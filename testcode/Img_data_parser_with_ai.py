import os
from langchain_openai import ChatOpenAI
import openai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# 객체 생성
client = openai.OpenAI()


# ✅ Base64 인코딩 함수 정의
def encode_image_to_base64(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"❌ 이미지 인코딩 중 오류 발생: {e}")


# 시스템 및 사용자 프롬프트 설정
system_prompt = """당신은 공지사항 정리하는 담당입니다. 당신의 임무는 주어진 이미지에 있는 글자들을 추출하여 정리하는 것입니다."""
user_prompt = """당신에게 주어진 이미지는 아주대학교 한 공지사항 정보입니다. 공지사항의 데이터를 텍스트로 정리하세요."""

# 폴더 경로 설정
folder_path = "/Users/ellie/Desktop/아주챗봇 Data"

# 폴더 내 이미지 파일 가져오기
image_path = "/Users/ellie/Desktop/아주챗봇 Data/인공지능융합학과 11기 1차 학생모집 설명회 & 선배들과의 대화.jpg"
base64_image = encode_image_to_base64(image_path)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지에서 핵심 정보를 뽑아줘."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ],
    max_tokens=1000,
)

# 응답 처리
for choice in response.choices:
    print(choice.message.content)
