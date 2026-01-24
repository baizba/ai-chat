import time
from contextlib import contextmanager

import structlog
from transformers import pipeline


@contextmanager
def measure_time():
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


class LLMService:
    def __init__(self):
        self.generator_pipeline = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", device=0)
        self.log = structlog.getLogger()

    def answer(self, question: str, context: str) -> str:
        final_prompt = f"""<|system|>
You are a helpful assistant. Answer only using the provided context.
This context is from the CV and describes professional experience of Branislav Vidovic.
If you can not find the answer then politely reply that you did not find the information that was asked.<|end|>
<|user|>
Context: {context}
Question: {question}<|end|>
<|assistant|>"""

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
