import torch

from ai_chat.llm_service import LLMService

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

context = """Classify the user question and return a single label.
Questions are about Branislav's resume.
Labels are:
FACT_SINGLE_ROLE
FACT_LIST
SUMMARY
EXPLANATION
OUT_OF_SCOPE"""
question = "What did Branislav do in IBM?"

llm_service = LLMService()
answer = llm_service.answer(question, context)
print(answer)
