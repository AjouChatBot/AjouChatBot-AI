import json
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from your_keyword_extractor import extract_keywords  # TODO
import os
from dotenv import load_dotenv

load_dotenv()

# Milvus 연결
connections.connect(host='localhost', port='19530') #TODO

# Milvus 컬렉션 스키마 정의
collection_name = "a_mate"
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="data_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="createAt", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="scrapUrl", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="urlTitle", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="dataType", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="fileName", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="fileIndex", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1000)
]

schema = CollectionSchema(fields, description="Embeddings with metadata")
collection = Collection(name=collection_name, schema=schema)

# OpenAI API 설정
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# JSON 데이터를 불러온다고 가정 (ex: API에서 받아오거나 로컬 파일 등)
def load_json_data(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 텍스트 청킹 및 임베딩
def process_json_item(item: dict):
    text = item.get("text", "")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    metadata_base = {
        "createAt": item.get("createAt", ""),
        "scrapUrl": item.get("scrapUrl", ""),
        "urlTitle": item.get("urlTitle", ""),
        "dataType": item.get("dataType", ""),
        "fileName": item.get("fileName", ""),
        "fileIndex": item.get("fileIndex", "")
    }

    insert_data = []
    for chunk in chunks:
        embedding = openai.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding
        keywords = extract_keywords(chunk)  # 예: ["python", "milvus"]
        record = [
            embedding,
            metadata_base["createAt"],
            metadata_base["scrapUrl"],
            metadata_base["urlTitle"],
            metadata_base["dataType"],
            metadata_base["fileName"],
            metadata_base["fileIndex"],
            ", ".join(keywords)
        ]
        insert_data.append(record)

    return insert_data

# 전체 JSON 처리 후 Milvus 삽입
def embed_all_json(json_path: str):
    raw_data = load_json_data(json_path)
    total_chunks = 0

    for item in raw_data:
        partition_name = item.get("category", "default").replace("-", "_")

        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)

        records, part = process_json_item(item), partition_name
        total_chunks += len(records)
        collection.insert(records, partition_name=part)

    collection.flush()
    print(f"{total_chunks} chunks inserted into Milvus.")

# 실행
if __name__ == "__main__":
    embed_all_json("your_data.json")  # TODO