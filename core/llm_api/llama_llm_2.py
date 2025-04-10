import asyncio
import collections
import json
import logging
import pathlib
import os
import re
import time
from pathlib import Path
from datetime import datetime
from traceback import format_exc
from typing import Optional, Union
from scipy.special import logsumexp

import aiohttp
import attrs
from transformers import AutoTokenizer
from anthropic import AsyncAnthropic
from anthropic.types.completion import Completion as AnthropicCompletion
from termcolor import cprint

from core.llm_api.base_llm import (
    PRINT_COLORS,
    LLMResponse,
    ModelAPIProtocol,
    messages_to_single_prompt,
    StopReason,
)
from core.llm_api.openai_llm import OAIChatPrompt

HUGGINGFACE_MODELS = {
    "meta-llama/Llama-3.2-1B": "https://gq139s723bs7rxgy.us-east-1.aws.endpoints.huggingface.cloud"
}

LOGGER = logging.getLogger(__name__)


def count_tokens(prompt: str) -> int:
    return len(prompt.split())


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    return 0, 0


@attrs.define()
class HuggingFaceModel(ModelAPIProtocol):
    def __init__(
        self,
        num_threads: int,
        token: str,
        prompt_history_dir: Path | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

        self.token = token
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.kwarg_change_name = {
            "temperature": "temperature",
            "max_tokens": "max_new_tokens",
            "top_p": "top_p",
            "logprobs": "top_n_tokens",
            "seed": "seed",
        }

        self.stop_reason_map = {
            "length": StopReason.MAX_TOKENS.value,
            "eos_token": StopReason.STOP_SEQUENCE.value,
        }

        self.tokenizers = {}

    def count_tokens(self, text, model_name):
        if model_name not in self.tokenizers:
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, use_auth_token=self.token)
        tokenizer = self.tokenizers[model_name]
        return len(tokenizer.tokenize(text))

    async def query(self, model_url, payload, session):
        async with session.post(model_url, headers=self.headers, json=payload) as response:
            return await response.json()

    async def run_query(self, model_url, input_text, params):
        payload = {
            "inputs": input_text,
            "parameters": params | {"details": True, "return_full_text": False},
        }
        async with aiohttp.ClientSession() as session:
            response = await self.query(model_url, payload, session)
            if "error" in response:
                # sometimes models break on specific inputs so this helps debug that
                print(response)
                LOGGER.error(response)
                if response["error"] == "Request failed during generation: Server error: CANCELLED":
                    if not pathlib.Path("error_input_text.txt").exists():
                        with open("error_input_text.txt", "w") as f:
                            f.write(payload["inputs"])

                raise RuntimeError(f"API Error: {response['error']}")
            return response

    @staticmethod
    def parse_logprobs(response_details) -> list[dict[str, float]] | None:
        if "top_tokens" not in response_details:
            return None

        logprobs = []
        for logprobs_at_pos in response_details["top_tokens"]:
            possibly_duplicated_logprobs = collections.defaultdict(list)
            for token_info in logprobs_at_pos:
                possibly_duplicated_logprobs[token_info["text"]].append(token_info["logprob"])

            logprobs.append({k: logsumexp(vs) for k, vs in possibly_duplicated_logprobs.items()})

        return logprobs

    async def __call__(
        self,
        model_ids: tuple[str, ...],
        prompt: Union[str, OAIChatPrompt, list],
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        truncate_prompt: bool = True,
        **kwargs,
    ) -> list[LLMResponse]:
        assert self.token is not None and self.token.startswith("hf_"), f"Invalid token {self.token}"

        start = time.time()
        assert len(model_ids) == 1, "Implementation only supports one model at a time."

        (model_id,) = model_ids
        assert model_id in HUGGINGFACE_MODELS, f"Model {model_id} not supported"
        model_url = HUGGINGFACE_MODELS[model_id]
        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        if isinstance(prompt, list):
            input_text = messages_to_single_prompt(prompt)
        elif hasattr(prompt, 'hf_format'):
            input_text = prompt.hf_format(hf_model_id=model_id)
        else:
            input_text = prompt

        if truncate_prompt and ("zephyr" in model_id or "Llama" in model_id):
            num_input_tokens = self.count_tokens(input_text, model_id) + 1
            if num_input_tokens > 3600:
                # truncate last user message to take away num_input_tokens - 3600 words
                user_message = input_text.split()[-1]
                num_tokens_to_remove = num_input_tokens - 3600
                num_words_to_remove = num_tokens_to_remove  # rough estimate
                new_user_message = " ".join(user_message.split()[:-num_words_to_remove])
                input_text = " ".join(input_text.split()[:-num_words_to_remove])
                print(f"Removed {num_words_to_remove} words from the last user message to reduce input length to 3600")

            old_max_tokens = kwargs.get("max_tokens", 2048)
            kwargs["max_tokens"] = min(4090 - num_input_tokens, old_max_tokens)
            if kwargs["max_tokens"] < old_max_tokens:
                print(
                    f"Reduced max_tokens from {old_max_tokens} to {kwargs['max_tokens']} for input length {num_input_tokens}"
                )

        # the circuit breaker model breaks on this specific prompt, so skip it
        if model_id == "meta-llama/Llama-3.2-1B" and "YOU ARE YOJA" in input_text:
            print("Skipping YOJA")
            response = LLMResponse(
                model_id=model_id,
                completion="Sorry, I cannot help with that.",
                stop_reason=StopReason.STOP_SEQUENCE.value,
                duration=0,
                api_duration=0,
                logprobs=None,
                cost=0,
            )
            return [response]

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        duration = None
        api_duration = None
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    params = {self.kwarg_change_name[k]: v for k, v in kwargs.items() if k in self.kwarg_change_name}
                    if "temperature" in kwargs:
                        if kwargs["temperature"] == 0:
                            # Huggingface API doesn't support temperature = 0
                            # Need to set do_sample = False instead
                            del params["temperature"]
                            params["do_sample"] = False
                        else:
                            params["do_sample"] = True
                    else:
                        params["do_sample"] = True
                        params["temperature"] = 1

                    response_data = await self.run_query(
                        model_url,
                        input_text,
                        params,
                    )
                    api_duration = time.time() - api_start
                    if not is_valid(response_data[0]["generated_text"]):
                        raise RuntimeError(f"Invalid response according to is_valid {response_data}")
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"

                if ("API Error: Input validation error: `inputs` must have less than" in str(e)) or (
                    "API Error: Input validation error: `inputs` tokens + `max_new_tokens` must be <=" in str(e)
                ):
                    # Early exit on context length errors.
                    raise e

                if "503 Service Unavailable" in str(e):
                    LOGGER.warn(f"503 Service Unavailable error. Waiting 60 seconds before retrying. (Attempt {i})")
                    await asyncio.sleep(60)
                else:
                    LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                    await asyncio.sleep(1.5**i)
            else:
                break

        if response_data is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        assert len(response_data) == 1, f"Expected length 1, got {len(response_data)}"

        response = LLMResponse(
            model_id=model_id,
            completion=response_data[0]["generated_text"],
            stop_reason=self.stop_reason_map[response_data[0]["details"]["finish_reason"]],
            duration=duration,
            api_duration=api_duration,
            logprobs=self.parse_logprobs(response_details=response_data[0]["details"]),
            cost=0,
        )
        responses = [response]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses

    def create_prompt_history_file(self, prompt, model_id, prompt_history_dir):
        """Create a file to store the prompt and response history."""
        if prompt_history_dir is None:
            return None
            
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        filepath = prompt_history_dir / filename
        
        with open(filepath, "w") as f:
            if isinstance(prompt, (str, dict)):
                json_str = json.dumps(prompt, indent=4)
            elif isinstance(prompt, list):
                # Handle list of messages by converting to single prompt
                json_str = json.dumps(messages_to_single_prompt(prompt), indent=4)
            else:
                # For objects with hf_format method (like OAIChatPrompt)
                json_str = json.dumps(prompt.hf_format(hf_model_id=model_id), indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)
        
        return filename

    def add_response_to_prompt_file(self, prompt_file, responses):
        """Add the response to the prompt history file."""
        if prompt_file is None or self.prompt_history_dir is None:
            return
            
        filepath = self.prompt_history_dir / prompt_file
        with open(filepath, "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            for response in responses:
                json_str = json.dumps(response.to_dict(), indent=4)
                json_str = json_str.replace("\\n", "\n")
                f.write(json_str)
                f.write("\n")