import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()

pairs = [
    ['Did he work at JIT?', 'He worked at JIT-Dienstleistungs GmbH, as Senior Java Developer and architect since March 2021 (ongoing employment)'],
    ['Did he work at Google?', 'He worked at JIT-Dienstleistungs GmbH, as Senior Java Developer and architect since March 2021 (ongoing employment)'],
    ['Did he work in IBM?', 'He worked at JIT-Dienstleistungs GmbH, as Senior Java Developer and architect since March 2021 (ongoing employment)']
]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)

def evaluate_candidates(question: str, candidates: list[str]):
    pass
