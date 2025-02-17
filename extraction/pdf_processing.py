import os
from pdf2image import convert_from_path
import numpy as np
from extraction.yolo_detection import process_image

def pdf_to_images(pdf_path, dpi=300): 
    """
    PDF 파일을 페이지별 이미지(PIL Image)로 변환합니다.
    
    :param pdf_path: PDF 파일 경로
    :param dpi: 변환 해상도 (기본값: 300)
    :return: PIL Image 객체 리스트
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"PDF를 {len(images)} 페이지 이미지로 변환하였습니다.")
    return images

def process_pdf(pdf_path, model, dpi=300):
    """
    PDF 파일을 페이지별 이미지로 변환한 후,
    각 페이지에서 YOLO 모델을 이용해 영역(박스)를 검출합니다.
    
    :param pdf_path: PDF 파일 경로
    :param model: 로드된 YOLO 모델
    :param dpi: 변환 해상도
    :return: (이미지 리스트, 모든 페이지의 검출 결과 리스트)
    """
    images = pdf_to_images(pdf_path, dpi=dpi)
    all_detections = []
    for page_number, image in enumerate(images, start=1):
        detections = process_image(image, model, page_number)
        all_detections.append(detections)
    return images, all_detections

def crop_detections(images, all_detections):
    """
    검출된 영역을 원본 이미지에서 크롭하고, 각 영역의 메타데이터를 생성합니다.
    
    :param images: 페이지별 원본 이미지 리스트
    :param all_detections: 각 페이지의 검출 영역 리스트
    :return: 영역 타입별로 분류된 딕셔너리 (예: "plain text", "table", "figure")
    """
    cropped_results = {"table": [], "plain text": [], "figure": []}
    for detections in all_detections:
        if not detections:
            continue
        page_number = detections[0]["page_number"]
        image = np.array(images[page_number - 1])
        for box in detections:
            x_min = int(box["x_min"])
            y_min = int(box["y_min"])
            x_max = int(box["x_max"])
            y_max = int(box["y_max"])
            cropped_img = image[y_min:y_max, x_min:x_max]
            category = box["class"]
            region_dict = {
                "unique_id": box["unique_id"],
                "image": cropped_img,
                "page_number": page_number,
                "bounding_box": {
                    "x_min": box["x_min"],
                    "y_min": box["y_min"],
                    "x_max": box["x_max"],
                    "y_max": box["y_max"]
                }
            }
            if category in cropped_results:
                cropped_results[category].append(region_dict)
            else:
                cropped_results.setdefault("other", []).append(region_dict)
    return cropped_results
