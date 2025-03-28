# rag_handler.py
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 기본 설정
index_name = "ajou-data"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
chat = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

def stream_rag_answer(query: str):
    retrieved_docs = vectorstore.similarity_search(query, k=10)
    docs_content = "\n---------------------------\n".join([doc.page_content for doc in retrieved_docs])

    prompt = PromptTemplate(
        template="""
당신은 아주대학교의 생활 도우미입니다. 다음 문서를 바탕으로 질문에 구체적으로 답하세요.
문서에 해당 내용이 없으면 모른다고 답하고, 단답형이 아닌 친절한 말투로 설명하세요.
문서:
{documents}

질문: {query}

답변:""",
        input_variables=["documents", "query"]
    )

    messages = [HumanMessage(content=prompt.format(documents=docs_content, query=query))]

    # Generator 형태로 응답 스트리밍
    def generator():
        for chunk in chat.stream(messages):
            if chunk.content:
                yield chunk.content

    return generator