import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()


def evaluate_employments(question: str, candidates: list[str]) -> list[float]:
    pairs: list[list[str]] = []
    for candidate in candidates:
        pairs.append([question, candidate])
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        assert len(scores) == len(pairs)  # if not then something is broken
        return [round(s, 2) for s in scores.tolist()]
