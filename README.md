# Chatbot Interface

This project provides a **FastAPI-based backend** that powers the chatbot on my portfolio website.  
It allows users to ask questions about my CV, with answers generated using a lightweight RAG pipeline.

The pipeline works as follows:
1. A structured, extended version of my CV is embedded.
2. The embeddings are stored in **ChromaDB**.
3. Incoming user questions are matched against these embeddings.
4. A local LLM generates the final answer using the retrieved context.

---

## 🚀 Features
- FastAPI REST endpoint for chat messages  
- CV content embedded and stored once in ChromaDB  
- Simple and efficient RAG flow  
- Fully local and privacy-friendly  

---

## 📡 API Endpoints

### **POST `/chat`**
Send a user message to the chatbot and get an AI-generated response.

#### **Request Body**
```json
{
  "message": "Your question goes here"
}
```
#### **Response example**
```json
{
  "question": "Your question goes here",
  "answer": "AI generated answer is here"
}
```