import os
import json
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# -------------- PDF 추출 관련 모듈 임포트 --------------
from extraction.pdf_processing import process_pdf, crop_detections
from extraction.yolo_detection import load_yolo_model
from extraction.ocr_extraction import process_plain_text_regions
from extraction.table_extraction import process_table_regions
from extraction.figure_extraction import process_figure_regions

# -------------- LLM 요약 생성 관련 설정 --------------
# 모델 이름 (gguf Q4 variant)
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# GPU 디바이스 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

def create_prompt_and_instruction(json_data):
    """
    json_data에서 '제목', '유형', '내용' 키를 활용해 요약 프롬프트와 instruction을 생성합니다.
    표와 도표의 경우, 제목은 별도로 지정합니다.
    """
    if json_data["region_type"] in ["일반표"]:
        title = "표 정보"
    elif json_data["region_type"] in ["도표"]:
        title = "도표 정보"
    else:
        title = json_data.get("제목", "문서 정보")
    
    category_prompts = {
        "차트": (
            "너는 차트를 분석하여 명확하고 객관적인 요약문을 생성하는 AI이다. "
            "아래의 차트 정보를 정리해서 하나의 문장으로 요약문을 생성하라. "
            "차근차근 생각해보자. 데이터에서 패턴, 공통점, 차이점, 이상치나 중요한 점이 있다면 이를 포함하라. "
            "입력 데이터 외의 정보를 추가로 추측하지 말아라."
        ),
        "표": (
            "너는 표를 분석하여 명확하고 객관적인 요약문을 생성하는 AI이다. "
            "아래의 표 정보를 정리해서 하나의 문장으로 요약문을 생성하라. "
            "주요 항목 간의 비교 및 공통점 혹은 차이점, 가장 두드러지는 부분도 서술하라. "
            "입력 데이터 외의 정보를 추가로 추측하지 말아라."
        )
    }
    
    if json_data["region_type"] == "일반표":
        prompt = category_prompts["표"]
    else:
        prompt = category_prompts["차트"]
    
    instruction = (
        f"다음은 {title}에 대한 설명입니다.\n"
        f"유형: {json_data['region_type']}\n"
        f"내용: {json_data['content']}\n"
        "위 내용을 기반으로 요약문을 작성해줘."
    )
    return prompt, instruction

def generate_summary(json_item):
    """
    입력 json_item(표 또는 도표 결과)에 대해 LLM을 사용하여 요약문을 생성하고 반환합니다.
    """
    prompt, instruction = create_prompt_and_instruction(json_item)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": instruction}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()

def main():
    # 사용자로부터 PDF 파일 경로 입력 받기
    pdf_path = input("PDF 파일 경로를 입력하세요: ").strip()
    
    # 1. YOLO 모델 로드 및 PDF 처리 (이미지 변환 및 영역 검출)
    yolo_model = load_yolo_model()
    images, all_detections = process_pdf(pdf_path, yolo_model, dpi=300)
    cropped_results = crop_detections(images, all_detections)
    
    # 2. 영역별(평문, 표, 도표) 분리
    plain_text_regions = cropped_results.get("plain text", [])
    table_regions = cropped_results.get("table", [])
    figure_regions = cropped_results.get("figure", [])
    
    print(f"평문 영역 수: {len(plain_text_regions)}")
    print(f"표 영역 수: {len(table_regions)}")
    print(f"도표 영역 수: {len(figure_regions)}")
    
    # 3. 영역별 텍스트 및 정보 추출
    plain_text_extraction_results = process_plain_text_regions(plain_text_regions)
    table_extraction_results = process_table_regions(table_regions)
    figure_extraction_results = process_figure_regions(figure_regions)
    
    # 4. 표와 도표 영역에 대해 LLM 요약 생성
    for result in table_extraction_results:
        try:
            summary = generate_summary(result)
            result["요약"] = summary
        except Exception as e:
            print(f"{result['data_id']} 표 요약 생성 실패: {e}")
            result["요약"] = result["content"]
    for result in figure_extraction_results:
        try:
            summary = generate_summary(result)
            result["요약"] = summary
        except Exception as e:
            print(f"{result['data_id']} 도표 요약 생성 실패: {e}")
            result["요약"] = result["content"]
    
    # 5. 모든 영역 결과 합치기 및 정렬 (페이지, y좌표, x좌표 순)
    combined_results = plain_text_extraction_results + table_extraction_results + figure_extraction_results
    combined_results.sort(key=lambda x: (x["page_number"],
                                           x["meta"]["bounding_box"]["y_min"],
                                           x["meta"]["bounding_box"]["x_min"]))
    
    # 6. 결과 저장 (JSON, CSV)
    RESULTS_DIR = "Output"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    final_combined_json = os.path.join(RESULTS_DIR, "result.json")
    with open(final_combined_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=4)
    print(f"JSON 결과 저장: {final_combined_json}")
    
    df = pd.DataFrame(combined_results)
    df["meta"] = df["meta"].apply(lambda m: json.dumps(m, ensure_ascii=False))
    final_combined_csv = os.path.join(RESULTS_DIR, "result.csv")
    df.to_csv(final_combined_csv, index=False, encoding='euc-kr')
    print(f"CSV 결과 저장: {final_combined_csv}")

if __name__ == "__main__":
    main()