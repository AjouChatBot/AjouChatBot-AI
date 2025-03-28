from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# 환경 변수 로드
load_dotenv()

# Pinecone 초기화
pinecone_client = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# 인덱스 이름
index_name = "ajou-data"

# Embedding 초기화
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# VectorStore 초기화
vectorstore = PineconeVectorStore(
    index_name=index_name,  # 인덱스 이름을 문자열로 전달
    embedding=embeddings,
)

# 검색 쿼리
# query = "수강신청 여석 잡고싶은데 어떻게해?"
query = input()

# 응답 시간 측정 시작
start_time = time.time()

# 유사도 검색
retrieved_documents = vectorstore.similarity_search(query, k=10)  # 상위 5개 문서 검색

# 결과를 활용하는 추가 로직
# 문서 내용을 하나로 병합 (예: ChatGPT로 질문 전달에 사용)
documents_content = "\n---------------------------\n".join(
    [doc.page_content for doc in retrieved_documents]
)

# Optional: ChatGPT로 전달할 프롬프트 준비

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = PromptTemplate(
    template="""당신은 아주대학교의 생활 도우미입니다. 다음 문서를 바탕으로 질문에 구체적으로 답하세요. 문서에 해당 내용이 포함되지 않은 경우 알 수 없다는 말을 전하세요. 
    단답식으로 답변하지 말고, 구체적으로 도우미 답게 친절한 말투로 답변하세요. 사용자에게 문서라는 단어를 언급하지 마세요.

문서:
{documents}

질문: {query}

답변:""",
    input_variables=["documents", "query"]
)

print("\nAI 답변:")
response = chat.stream([
    HumanMessage(content=prompt.format(documents=documents_content, query=query))
])

for token in response:
    print(token.content, end="", flush=True)

# 응답 시간 측정 종료
end_time = time.time()

# 응답 시간 출력
print(f"\n\n응답 시간: {end_time - start_time:.2f}초")
