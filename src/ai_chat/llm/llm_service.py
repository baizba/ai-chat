import time
from contextlib import contextmanager

import structlog
from transformers import pipeline

from ai_chat.service import prompts


@contextmanager
def measure_time():
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


def get_final_prompt(prompt_template: str, context: str, question: str) -> str:
    return prompt_template.replace("${context}", context).replace("${question}", question)


class LLMService:
    def __init__(self):
        self.generator_pipeline = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", device_map="auto", dtype="auto")
        self.log = structlog.getLogger()

    def answer_general(self, question: str, context: str) -> str:
        final_prompt = get_final_prompt(prompts.general_prompt, context, question)

        with measure_time() as elapsed_time:
            response = self.generator_pipeline(final_prompt)

        if len(response) == 0:
            return "no_answer"

        generated_text = response[0]["generated_text"]
        if generated_text.find("<|assistant|>") == -1:
            return "no_answer"

        final_response = generated_text.split("<|assistant|>")[1].strip()
        self.log.info(
            "llm.service.response",
            in_tokens=len(self.generator_pipeline.tokenizer.encode(final_prompt)),
            out_tokens=len(self.generator_pipeline.tokenizer.encode(final_response)),
            model=self.generator_pipeline.model.config.name_or_path,
            duration_ms=round(elapsed_time())
        )

        return final_response

    def answer(self, prompt_template: str, question: str, context: str) -> str:
        final_prompt = get_final_prompt(prompt_template, context, question)

        with measure_time() as elapsed_time:
            response = self.generator_pipeline(final_prompt)

        if len(response) == 0:
            return "no_answer"

        generated_text = response[0]["generated_text"]
        if generated_text.find("<|assistant|>") == -1:
            return "no_answer"

        final_response = generated_text.split("<|assistant|>")[1].strip()
        self.log.info(
            "llm.service.response",
            in_tokens=len(self.generator_pipeline.tokenizer.encode(final_prompt)),
            out_tokens=len(self.generator_pipeline.tokenizer.encode(final_response)),
            model=self.generator_pipeline.model.config.name_or_path,
            duration_ms=round(elapsed_time())
        )

        return final_response
