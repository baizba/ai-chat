from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from LocalEmbeddings import LocalEmbeddings
from models import ChatRequest
from models import ChatResponse

app = FastAPI()

# start end execute embedding of local texts
local_embeddings = LocalEmbeddings()
local_embeddings.perform_embeddings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
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


@app.post("/chat", response_model=ChatResponse)
async def get_hello(request: ChatRequest):
    # sentence = random.choice(sentences)
    return local_embeddings.perform_query(request.message)
