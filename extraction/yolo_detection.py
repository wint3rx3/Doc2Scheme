import numpy as np
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
def load_yolo_model():
    """
    DocLayout-YOLO 모델을 Hugging Face Hub에서 다운로드하여 로드합니다.
    
    :return: YOLOv10 모델 객체
    """
    filepath = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    return YOLOv10(filepath)

def calculate_iou(box1, box2):
    """
    두 박스 간 Intersection over Union (IoU)를 계산합니다.
    
    :param box1: (x1, y1, x2, y2)
    :param box2: (x3, y3, x4, y4)
    :return: IoU 값
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    return inter_area / (box1_area + box2_area - inter_area)

def filter_duplicate_boxes(bounding_boxes, iou_threshold=0.5):
    """
    중복된 박스를 IoU 기준으로 제거합니다.
    
    :param bounding_boxes: 검출된 박스 리스트
    :param iou_threshold: IoU 임계값 (기본값: 0.5)
    :return: 중복 제거된 박스 리스트
    """
    filtered_boxes = []
    for box in bounding_boxes:
        keep = True
        for fbox in filtered_boxes:
            iou = calculate_iou(
                (box["x_min"], box["y_min"], box["x_max"], box["y_max"]),
                (fbox["x_min"], fbox["y_min"], fbox["x_max"], fbox["y_max"])
            )
            if iou > iou_threshold:
                if box["confidence"] > fbox["confidence"]:
                    filtered_boxes.remove(fbox)
                else:
                    keep = False
                break
        if keep:
            filtered_boxes.append(box)
    return filtered_boxes

def generate_unique_suffix(index):
    """
    고유한 접미사를 생성합니다.
    
    :param index: 정수 인덱스
    :return: 알파벳 한 글자
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return alphabet[index % len(alphabet)]

def process_image(image, model, page_number):
    """
    단일 페이지 이미지에서 YOLO 모델을 사용해 영역을 검출합니다.
    
    :param image: PIL 이미지
    :param model: YOLO 모델
    :param page_number: 페이지 번호
    :return: 검출된 박스 리스트 (중복 제거 포함)
    """
    image_array = np.array(image)
    # 모델 예측 (DEVICE: GPU 또는 CPU)
    det_res = model.predict(image_array, imgsz=1024, conf=0.2, device=DEVICE)
    bounding_boxes = []
    for i, box in enumerate(det_res[0].boxes):
        class_name = model.names[int(box.cls)]
        class_number = int(box.cls)
        unique_suffix = generate_unique_suffix(i)
        bounding_boxes.append({
            "class": class_name,
            "confidence": float(box.conf),
            "x_min": float(box.xyxy[0][0]),
            "y_min": float(box.xyxy[0][1]),
            "x_max": float(box.xyxy[0][2]),
            "y_max": float(box.xyxy[0][3]),
            "unique_id": f"page{page_number}_class{class_number}_{unique_suffix}",
            "page_number": page_number
        })
    filtered_boxes = filter_duplicate_boxes(bounding_boxes, iou_threshold=0.5)
    return filtered_boxes
