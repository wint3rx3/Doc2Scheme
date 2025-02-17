import os
import json
import pandas as pd

# 모듈 import
from extraction.pdf_processing import process_pdf, crop_detections
from extraction.yolo_detection import load_yolo_model
from extraction.ocr_extraction import process_plain_text_regions
from extraction.table_extraction import process_table_regions
from extraction.figure_extraction import process_figure_regions

def main():
    # PDF 파일 경로 (실제 파일 경로로 수정)
    pdf_path = "비타민 CV 프로젝트.pdf"
    
    # 1. YOLO 모델 로드 및 PDF 처리 (이미지 변환 및 영역 검출)
    model = load_yolo_model()  # DocLayout-YOLO 모델 로드
    images, all_detections = process_pdf(pdf_path, model, dpi=300)
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
    
    # 4. 모든 영역 결과 합치기 및 정렬 (페이지, y좌표, x좌표 순)
    combined_results = plain_text_extraction_results + table_extraction_results + figure_extraction_results
    combined_results.sort(key=lambda x: (x["page_number"],
                                           x["meta"]["bounding_box"]["y_min"],
                                           x["meta"]["bounding_box"]["x_min"]))
    
    # 5. 결과 저장 (JSON, CSV)
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
