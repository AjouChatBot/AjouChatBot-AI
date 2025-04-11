# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_handler import stream_rag_answer, get_chat_history

app = FastAPI()

class Question(BaseModel):
    user_id: str
    question: str
    is_new_topic: bool

@app.post("/chat")
async def ask_question(payload: Question):
    user_id = payload.user_id
    question = payload.question
    is_new_topic = payload.is_new_topic

    stream_generator = await stream_rag_answer(user_id, question, is_new_topic)

    return StreamingResponse(
        content=stream_generator(),
        media_type="text/plain"
    )

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    history = await get_chat_history(user_id)
    return history