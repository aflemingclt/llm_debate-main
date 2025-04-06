import asyncio
import collections
import json
import logging
import pathlib
import os
import re
import time
import random
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


async def count_tokens(text: str, model_id: str) -> int:
    """Count tokens using the model's tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return len(tokenizer.encode(text))


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    return 0, 0


@attrs.define()
class HuggingFaceModel(ModelAPIProtocol):
    num_threads: int
    token: str
    prompt_history_dir: Path
    print_prompt_and_response: bool = False
    available_requests: asyncio.BoundedSemaphore = attrs.field(init=False)
    session: aiohttp.ClientSession = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))
        self.session = aiohttp.ClientSession(headers={
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        })
        self.prompt_history_dir.mkdir(exist_ok=True)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    @staticmethod
    def _create_prompt_history_file(prompt):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        with open(os.path.join("prompt_history", filename), "w") as f:
            json_str = json.dumps(prompt, indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, response):
        with open(os.path.join("prompt_history", prompt_file), "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps(response.to_dict(), indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, OAIChatPrompt],
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert len(model_ids) == 1, "HuggingFace implementation only supports one model at a time."
        model_id = model_ids[0]
        if isinstance(prompt, list):
            prompt = messages_to_single_prompt(prompt)

        # Format the request payload according to HF API requirements
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get('temperature', 1.0),
                "top_p": kwargs.get('top_p', 1.0),
                "max_new_tokens": kwargs.get('max_tokens', 1000),
                "do_sample": True,
                "return_full_text": False  # Only return the generated text, not the prompt
            }
        }

        prompt_file = self._create_prompt_history_file(prompt)
        LOGGER.debug(f"Making {model_id} call")
        response_text = None
        api_duration = None

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    
                if i > 0:
                    base_delay = min(300, 1.5 ** i)
                    jitter = random.uniform(0, 0.1 * base_delay)
                    retry_delay = base_delay + jitter
                    LOGGER.info(f"Waiting {retry_delay:.2f} seconds before retry {i+1}/{max_attempts}")
                    await asyncio.sleep(retry_delay)
                
                timeout = aiohttp.ClientTimeout(total=30)
                async with self.session.post(
                    HUGGINGFACE_MODELS[model_id],
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        # Handle the response format based on HF API
                        if isinstance(response_json, list) and len(response_json) > 0:
                            response_text = response_json[0].get('generated_text', '')
                        else:
                            response_text = response_json.get('generated_text', '')
                        api_duration = time.time() - api_start
                        break
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"API returned status code {response.status}: {error_text}")
                            
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                if i + 1 < max_attempts:
                    LOGGER.warning(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i+1}/{max_attempts})")
                else:
                    LOGGER.error(f"Final attempt failed: {error_info}")
                    raise

        if response_text is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        num_context_tokens = await count_tokens(prompt, model_id)
        num_completion_tokens = await count_tokens(response_text, model_id)
        
        context_token_cost, completion_token_cost = price_per_token(model_id)
        cost = (
            num_context_tokens * context_token_cost
            + num_completion_tokens * completion_token_cost
        )
        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        llm_response = LLMResponse(
            model_id=model_id,
            completion=response_text,
            stop_reason="max_tokens",
            duration=duration,
            api_duration=api_duration,
            cost=cost,
        )

        self._add_response_to_prompt_file(prompt_file, llm_response)
        if self.print_prompt_and_response or print_prompt_and_response:
            cprint(prompt, "yellow")
            pattern = r"(Human: |Assistant: )(.*?)(?=(Human: |Assistant: )|$)"
            for match in re.finditer(
                pattern, prompt, re.S
            ):  # re.S makes . match any character, including a newline
                role = match.group(1).removesuffix(": ").lower()
                role = {"human": "user"}.get(role, role)
                cprint(match.group(2), PRINT_COLORS[role])
            cprint(f"Response ({llm_response.model_id}):", "white")
            cprint(
                f"{llm_response.completion}", PRINT_COLORS["assistant"], attrs=["bold"]
            )
            print()

        return [llm_response]
