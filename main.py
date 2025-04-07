# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_handler import stream_rag_answer

app = FastAPI()

class Question(BaseModel):
    user_id: str
    question: str
    is_new_topic: bool

@app.post("/api/v0/chat")
async def ask_question(payload: Question):
    user_id = payload.user_id
    question = payload.question
    is_new_topic = payload.is_new_topic

    stream_generator = stream_rag_answer(user_id, question, is_new_topic)

    return StreamingResponse(
        content=stream_generator(),
        media_type="text/plain"
    )