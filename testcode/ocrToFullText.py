import json

def extract_text_from_vision_json(json_path: str) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result_text = []

    pages = data.get("fullTextAnnotation", {}).get("pages", [])
    for page in pages:
        for block in page.get("blocks", []):
            block_text = []
            for paragraph in block.get("paragraphs", []):
                para_text = []
                for word in paragraph.get("words", []):
                    word_text = ""
                    for symbol in word.get("symbols", []):
                        word_text += symbol.get("text", "")
                        # 줄바꿈 또는 띄어쓰기 정보
                        break_type = symbol.get("property", {}).get("detectedBreak", {}).get("type")
                        if break_type == "SPACE":
                            word_text += " "
                        elif break_type == "LINE_BREAK":
                            word_text += "\n"
                    para_text.append(word_text.strip())
                block_text.append(" ".join(para_text).strip())
            result_text.append("\n".join(block_text).strip())

    final_text = "\n\n".join(result_text).strip()  # 블록마다 줄바꿈 2번으로 구분
    return final_text


# 사용 예시

path = "../1.json"  # 또는 절대 경로
clean_text = extract_text_from_vision_json(path)
print(clean_text)  # 혹은 벡터 DB 임베딩에 전달