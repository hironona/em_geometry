
from typing import Dict, List
import math
from functools import lru_cache
from pathlib import Path
import yaml
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client (uses OPENAI_API_KEY env variable by default)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Default OpenAI model to use for judging
DEFAULT_MODEL = "gpt-4o"


class OpenAiJudge:
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""

    def __init__(self, model: str = DEFAULT_MODEL, prompt_template: str = None):
        self.model = model
        self.prompt_template = prompt_template


    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )

        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))

        return result

    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explicitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
