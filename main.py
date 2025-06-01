# main.py
from fastapi import HTTPException
from typing import List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from rag_handler_milvus import stream_rag_answer, get_chat_history
from extract_topic import extract_topics_from_text

app = FastAPI()

class Question(BaseModel):
    user_id: str
    question: str
    is_new_topic: bool
    keywords: List[str]

class TextInput(BaseModel):
    text: str

@app.post("/chat")
async def ask_question(payload: Question):
    if not payload.user_id or not payload.question:
        raise HTTPException(status_code=400, detail="user_id and question are required.")

    user_id = payload.user_id
    question = payload.question
    is_new_topic = payload.is_new_topic
    keywords = payload.keywords

    stream_generator = await stream_rag_answer(user_id, question, is_new_topic)

    return StreamingResponse(
        content=stream_generator(),
        media_type="text/plain"
    )

@app.post("/subject")
async def extract_topic(input: TextInput):
    return await extract_topics_from_text(input.text)

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    history = await get_chat_history(user_id)
    return history

@app.get("/health")
async def check_health():
    return PlainTextResponse(content="ok", status_code=200)