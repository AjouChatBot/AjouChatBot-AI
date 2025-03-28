import os
import easyocr

# 이미지가 있는 폴더 경로
folder_path = "/Users/ellie/Desktop/아주챗봇 Data"

# 지원하는 이미지 확장자
image_extensions = (".png", ".jpg", ".jpeg")
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

if not image_files:
    print("이미지 파일이 존재하지 않습니다.")
else:
    image_path = os.path.join(folder_path, image_files[0])
    print(f"선택된 이미지 파일: {image_path}")

    def extract_text_from_image(image_path):
        """ EasyOCR을 이용해 한글, 영어, 숫자 OCR 실행 """
        reader = easyocr.Reader(['ko', 'en'])  # 한글(kor)과 영어(eng) 지원
        result = reader.readtext(image_path, detail=0)  # 텍스트만 반환
        return "\n".join(result)

    # 텍스트 추출
    text = extract_text_from_image(image_path)
    print("추출된 텍스트:\n", text)