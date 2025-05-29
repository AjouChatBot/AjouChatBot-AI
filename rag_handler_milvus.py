# rag_handler.py
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import json
import redis.asyncio as aioredis

load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')

redis_client = aioredis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

MAX_MESSAGES = 6

# Milvus 설정
collection_name = "a_mate"
connection_args = {
    "host": MILVUS_HOST,
    "port": MILVUS_PORT
}

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Milvus(
    collection_name=collection_name,
    embedding_function=embeddings,
    connection_args=connection_args,
    # partition_name="Facilities"
)

chat = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
chat_histories = {}

# 채팅 이력 가져오기
async def get_chat_history(user_id: str):
    data = await redis_client.lrange(f"chat_history:{user_id}", 0, -1)
    return [json.loads(item) for item in data]

# 채팅 이력 저장
async def add_to_chat_history(user_id: str, message):
    message_data = {
        "type": "human" if isinstance(message, HumanMessage) else "ai",
        "content": message.content
    }
    await redis_client.rpush(f"chat_history:{user_id}", json.dumps(message_data))
    await redis_client.ltrim(f"chat_history:{user_id}", -MAX_MESSAGES, -1)

# 메시지 파싱
def parse_message(data):
    if data["type"] == "human":
        return HumanMessage(content=data["content"])
    return AIMessage(content=data["content"])


# RAG 응답 생성
async def stream_rag_answer(user_id: str, query: str, is_new_topic: bool):
    if is_new_topic:
        await redis_client.delete(f"chat_history:{user_id}")
        history = []
    else:
        raw_history = await get_chat_history(user_id)
        history = [parse_message(msg) for msg in raw_history]

    # 유사 문서 검색 (중복 제거)
    raw_docs = vectorstore.similarity_search(query, k=30)
    unique_docs = []
    seen_ids = set()

    for doc in raw_docs:
        doc_id = doc.metadata.get("id")  # "id"는 실제 필드 이름에 맞게 수정
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
        if len(unique_docs) == 10:
            break

    docs_content = "\n---------------------------\n".join([
        f"[출처: {doc.metadata.get('urlTitle', '제목 없음')}] {doc.page_content}\n링크: {doc.metadata.get('scrapUrl', '링크 없음')}"
        for doc in unique_docs
    ])

    # Prompt 구성
    prompt = PromptTemplate(
        template="""
당신은 아주대학교의 생활 도우미 챗봇입니다. 사용자의 질문에 답할 때는 반드시 다음을 지키세요:

1. 문서에 정보가 없는 경우 또는 문서가 비어 있는 경우, 답변하지 말고 다음과 같이 정중하게 안내하세요:
    - 죄송합니다. 현재 관련 정보를 찾을 수 없어요.
    - 이 내용은 제가 참고할 수 있는 문서에 포함되어 있지 않아서 정확히 답변드리기 어려워요.
    - 현재는 해당 정보를 제공해드릴 수 없습니다.
    무리하게 추측하거나 일반적인 내용을 말하지 말고, 명확하게 정보 부족 상황을 안내하세요.
2. 사용자의 질문에 답하기 위해 추가 정보가 필요한 경우, 그 정보를 요청하세요.
    예를 들어, 학사 제도나 졸업 요건 등은 학번, 학과, 학년 등에 따라 달라질 수 있으므로, 다음과 같이 자연스럽게 되물어보세요: 
   - 정확한 안내를 위해 몇 학번이신지 알려주실 수 있나요?" 
   - 소속 학과나 전공도 함께 알려주시면 더 정확한 정보를 드릴 수 있어요.
   단, 모든 질문에 대해 무조건 학번 및 학과를 묻지 말고, 그 정보가 실제로 필요한 경우에만 물어보세요.
3. 문서에 근거하지 않고 임의로 추측하거나 일반적인 기준만 제시하지 마세요. 필요한 정보를 받은 이후에, 문서를 바탕으로 구체적으로 답변하세요.
4. 사용자가 이전 답변에 대해 후속 질문을 할 경우, 반드시 이전 문맥을 반영해서 자연스럽게 이어서 대답하세요.
5. 답변에 따옴표("")를 사용하지 마세요. 텍스트를 인용처럼 표시하지 말고, 자연스러운 설명체로만 답변하세요.

문서:
{documents}

질문: {query}

답변:""",
        input_variables=["documents", "query"]
    )

    user_prompt = prompt.format(documents=docs_content, query=query)
    messages = history[-MAX_MESSAGES:] + [HumanMessage(content=user_prompt)]

    async def generator():
        response_text = ""
        for chunk in chat.stream(messages):
            if chunk.content:
                response_text += chunk.content
                yield chunk.content

        # 응답 후 채팅 저장
        await add_to_chat_history(user_id, HumanMessage(content=query))
        await add_to_chat_history(user_id, AIMessage(content=response_text))

    return generator
