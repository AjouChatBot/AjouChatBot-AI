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

# Redis 설정
redis_client = aioredis.Redis(host='localhost', port=6379, decode_responses=True)
MAX_MESSAGES = 6

load_dotenv()

# Milvus 설정
collection_name = "a_mate"
connection_args = {"host": "mate.ajou.app", "port": "28116"}

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

    # 유사 문서 검색
    retrieved_docs = vectorstore.similarity_search(query, k=10)
    docs_content = "\n---------------------------\n".join([
        f"[출처: {doc.metadata.get('urlTitle', '제목 없음')}] {doc.page_content}\n링크: {doc.metadata.get('scrapUrl', '링크 없음')}"
        for doc in retrieved_docs
    ])

    # Prompt 구성
    prompt = PromptTemplate(
        template="""
당신은 아주대학교의 생활 도우미 챗봇입니다. 사용자의 질문에 답할 때는 반드시 다음을 지키세요:

1. 문서에서 답변을 찾을 수 없거나 질문에 정보가 부족한 경우, **질문에 필요한 정보를 되묻고**, 충분한 정보가 주어진 후에만 답변하세요.
2. 예를 들어 '졸업 총 이수학점이 몇 점인가요?' 라는 질문이 오면, 학번이나 학과, 학년 등의 정보가 없다면 즉시 질문하지 말고 다음과 같이 물어보세요:  
   - "몇 학번이신가요?"  
   - "소속 학과도 함께 알려주시면 정확한 정보를 드릴 수 있어요."
3. 절대 임의로 추측하거나 일반적인 기준을 말하지 말고, 필요한 정보를 받은 후에만 문서를 바탕으로 구체적으로 답하세요.
4. 사용자가 이전 답변에 대해 후속 질문을 하면 이전 문맥을 반영해서 대답하세요.

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
