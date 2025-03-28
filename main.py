# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_handler import stream_rag_answer

app = FastAPI()

class Question(BaseModel):
    user_id: str
    question: str

@app.post("/api/v0/chat")
async def ask_question(payload: Question):
    question = payload.question

    stream_generator = stream_rag_answer(question)

    return StreamingResponse(
        content=stream_generator(),
        media_type="text/plain"
    )