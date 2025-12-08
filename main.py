import random

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from ChatRequest import ChatRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # or ["http://localhost:4200"] for tighter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentences = [
    "Sounds interesting, i'll think about it.",
    "So what. Me too.",
    "No way. You're kidding.",
    "Well... It happens.",
    "How cool is that!"
]


@app.post("/chat")
async def get_hello(request: ChatRequest):
    sentence = random.choice(sentences)
    return {"question": request.message, "answer": sentence}
