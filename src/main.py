import torch

from src.llm_service import LLMService

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

context = "Branislav knows Java, Python and Huggingface. He likes to play Basketball."
question = "What does Branislav know and what sports does he like?"

llm_service = LLMService()
answer = llm_service.answer(question, context)
print(answer)
