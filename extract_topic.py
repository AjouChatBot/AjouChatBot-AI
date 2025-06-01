import openai
from openai import OpenAI
from fastapi import HTTPException

client = OpenAI()

async def extract_topics_from_text(text: str) -> dict:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required.")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "다음 텍스트가 어떤 주제에 대한 것인지 한 문장으로 간결하게 요약하거나 제목처럼 표현해줘. 문장을 명사형으로 끝내고 마침표는 찍지 말아줘."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        extracted = response.choices[0].message.content
        return {"subject": extracted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))