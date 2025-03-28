import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from langchain.text_splitter import SpacyTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import spacy
from pinecone import Pinecone
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))  # None이면 .env 파일을 못 읽고 있는 것

# 🔹 일반 텍스트 입력
your_text = """
아주대학교 내에는 성호관, 팔달관, 일신관, 다산관에 편의점이 있어.
"""

# 🔹 일반 텍스트를 Document 객체로 변환
documents = [Document(page_content=your_text, metadata={"source": "manual_input"})]

print(f"✅ 일반 텍스트 1개를 문서로 변환 완료!")

# 🔹 Spacy 모델 로드 및 문서 분할
nlp = spacy.load("ko_core_news_sm")
nlp.max_length = 2_000_000

text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ko_core_news_sm",
    max_length=2000000,
    separator="\n\n"
)
splitted_documents = text_splitter.split_documents(documents)

print(f"✅ 일반 텍스트가 총 {len(splitted_documents)} 개의 청크로 분할되었습니다.")

# 🔹 OpenAI Embeddings 초기화
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 🔹 Pinecone 초기화
pinecone_client = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# 🔹 Pinecone VectorStore 초기화 및 문서 추가
index_name = "ajou-data"
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=index_name
)

vectorstore.add_documents(splitted_documents)  # 🔹 모든 문서 한 번에 업로드

print("✅ 일반 텍스트를 Pinecone 데이터베이스에 저장 완료!")