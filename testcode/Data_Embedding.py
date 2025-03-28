from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, \
    UnstructuredCSVLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import spacy
from pinecone import Pinecone
from langchain.schema import Document

import os
import olefile
import pandas as pd
import docx2txt
from dotenv import load_dotenv
import shutil

load_dotenv()

# 🔹 1. 여러 개의 문서 처리하기
data_directory = "/Users/ellie/Desktop/아주챗봇 Data"  # 데이터가 있는 폴더 경로
max_files = 10  # 처리할 최대 파일 개수
documents = []


# 🔹 2. 파일 확장자별 로드 함수
def load_hwp(file_path):
    """HWP 파일을 OLE 방식으로 읽어서 텍스트 추출"""
    try:
        with olefile.OleFileIO(file_path) as hwp:
            if hwp.exists('HwpSummaryInformation'):
                meta = hwp.get_metadata()
                text = meta.summary if meta.summary else "텍스트를 추출할 수 없습니다."
                return [Document(page_content=text, metadata={"source": file_path})]
            else:
                return [Document(page_content="해당 HWP 파일에서 텍스트를 추출할 수 없습니다.", metadata={"source": file_path})]
    except Exception as e:
        return [Document(page_content=f"HWP 파일 읽기 실패: {e}", metadata={"source": file_path})]


def load_docx(file_path):
    """DOCX 파일 로드"""
    text = docx2txt.process(file_path)
    return [Document(page_content=text, metadata={"source": file_path})]


def load_xlsx(file_path):
    """XLSX 파일 로드"""
    df = pd.read_excel(file_path)
    text = df.to_string()  # 엑셀 데이터를 문자열로 변환
    return [Document(page_content=text, metadata={"source": file_path})]


def load_csv(file_path):
    """CSV 파일 로드"""
    df = pd.read_csv(file_path)
    text = df.to_string()  # CSV 데이터를 문자열로 변환
    return [Document(page_content=text, metadata={"source": file_path})]


def load_pdf(file_path):
    """PDF 파일 로드"""
    loader = PyMuPDFLoader(file_path)
    pdf_documents = loader.load()

    # PyMuPDFLoader는 이미 Document 객체 리스트를 반환하므로 그대로 반환
    return pdf_documents


# 🔹 3. 파일 확장자별로 문서 로드
file_list = os.listdir(data_directory)[:max_files]

for filename in file_list:
    file_path = os.path.join(data_directory, filename)

    if filename.endswith(".pdf"):
        documents.extend(load_pdf(file_path))
    elif filename.endswith(".hwp"):
        documents.extend(load_hwp(file_path))
    elif filename.endswith(".docx"):
        documents.extend(load_docx(file_path))
    elif filename.endswith(".xlsx"):
        documents.extend(load_xlsx(file_path))
    elif filename.endswith(".csv"):
        documents.extend(load_csv(file_path))

print(f"총 {len(documents)}개의 문서를 로드했습니다.")


# 🔹 4. 문서 분할
# Spacy 모델 로드 및 max_length 설정
nlp = spacy.load("ko_core_news_sm")
nlp.max_length = 2_000_000

text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ko_core_news_sm",
    max_length = 2000000,
    separator="\n\n"
)
splitted_documents = text_splitter.split_documents(documents)

print(f"총 {len(splitted_documents)}개의 청크로 분할되었습니다.")

# 🔹 5. OpenAI Embeddings 초기화
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 🔹 6. Pinecone 초기화
pinecone_client = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# 🔹 7. Pinecone VectorStore 초기화 및 문서 추가
index_name = "ajou-data"
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=index_name,
    namespace="your_namespace"
)

vectorstore.add_documents(splitted_documents)  # 🔹 모든 문서 한 번에 업로드

print("✅ 모든 문서를 Pinecone 데이터베이스에 저장 완료!")


# 8. 완료 문서 이동
processed_data_directory = "/Users/ellie/Desktop/Done"  # 이동할 폴더 경로

if not os.path.exists(processed_data_directory):
    os.makedirs(processed_data_directory)  # 폴더 없으면 생성

for filename in file_list:
    file_path = os.path.join(data_directory, filename)
    new_path = os.path.join(processed_data_directory, filename)

    shutil.move(file_path, new_path)  # 파일 이동
    print(f"📂 파일 이동 완료: {filename} → {new_path}")
