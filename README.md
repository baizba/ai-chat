# Modular RAG System with Intent Routing and Ambiguity Handling

This project implements a modular Retrieval-Augmented Generation (RAG) system designed to answer questions about a structured CV.

The focus is not just retrieval, but improving answer reliability through:
- intent classification
- ambiguity handling
- constrained context for LLMs

## Design Decisions

- **Intent routing before retrieval**  
  Improves precision by narrowing the search space and applying domain-specific logic.

- **Limited context to LLM**  
  Only relevant CV chunks are passed to reduce noise and prevent hallucination.

- **Domain-specific handlers**  
  Each domain (employment, skills, etc.) applies tailored logic instead of relying solely on embeddings.

- **Ambiguity handling**  
  The system avoids answering unclear queries instead of guessing.

---

## The pipeline workflow:
1. A structured, extended version of my CV is chunked and embedded into **Chroma**.
2. Examples of user questions are embedded and stored in separate collection as intents.
3. Incoming user questions are matched against these intents to find matching domain.
4. If a matching domain is detected above a confidence threshold, the query is routed to the corresponding handler.
5. Question is handled by domain specific handler (skills, employment, profile etc.).
6. Each of the handlers can perform specific heuristics to detect specific parts of CV
7. This part of CV is then sent to LLM with domain specific prompt to summarize.
8. The system currently focuses on single-intent queries. Ambiguous and multi-part questions are intentionally not handled to maintain answer reliability.

---

## 🚀 Features
- FastAPI REST endpoint for chat messages and indexing
- CV content embedded and stored once in ChromaDB
- Logging with structlog events for observability
- Fully local and privacy-friendly
- Containerized and can run anywhere with docker
- End-to-end tests with question examples

---

## Limitations

- Complex multi-part questions are not handled
- Performance depends on embedding quality
- Temporal reasoning is rule-based and limited

---

## Architecture
- [CV Indexing](./src/ai_chat/indexing/cv_indexing_service.py) - CV embedding
- [Intent Classification](./src/ai_chat/intent/intent_classifier.py) - Intent embedding and search
- [Routing Logic](./src/ai_chat/router/query_router.py) - route query to handler based on intent
- [CV Chunk Retrieval](./src/ai_chat/retrieval/cv_service.py) - retrieve chunks of CV
- [Service Layer](./src/ai_chat/service) - Services for handling domain questions
- [LLM Layer](./src/ai_chat/llm/llm_service.py) - LLM service for interaction with LLM
- [Reranker](./src/ai_chat/llm/reranker.py) - Reranker to check if the answer satisfies the question and infer next steps
- [Architecture](./c4_architecture.drawio.svg) - Architecture C4
- [End-to-end Tests](./tests/rag_integration_test.py) - End-to-end tests with questions and answers

---

## 📡 Swagger
[Link to Swagger is here](http://localhost:8100/docs) - if you run it on some other server replace localhost

---

## Example Queries

- "What companies did he work for?"
- "Where did he work between 2018 and 2022?"
- "What did he do at Netconomy?"

Example response:
JIT-Dienstleistungs GmbH, ecx.io - IBM Company, Netconomy GmbH, Codecentric doo

---

## Testing Approach

The system is validated using end-to-end tests with paraphrased queries and edge cases (e.g. temporal queries, unknown entities).
The goal is to ensure consistent behavior across different phrasings and to detect failure modes early.

---

## Running

### Locally for debugging
1. Start chroma in the docker-compose-yaml
2. Run this in the project dir: $env:PYTHONPATH="src"; uvicorn ai_chat.server:app --reload --port 8100

### Everything with docker
This mode is convenient to run all in cloud: **docker compose up** in the ai-chat project

When starting first time open the swagger docs. Perform Indexing of the Intents and CV.
Then you can use the chat endpoint to ask questions.