import cv2
from PIL import Image
import torch
from transformers import Pix2StructForConditionalGeneration, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

class FigureExtractor:
    def __init__(self):
        # Pix2Struct 모델과 프로세서를 로드합니다.
        self.processor = AutoProcessor.from_pretrained("brainventures/deplot_kr")
        self.model = Pix2StructForConditionalGeneration.from_pretrained("brainventures/deplot_kr")
        self.model.to(DEVICE)
    
    def extract_figure_info(self, image):
        """
        도표 이미지에서 설명 텍스트를 생성합니다.
        
        :param image: 도표 영역 이미지 (numpy array 또는 PIL Image)
        :return: 생성된 도표 설명 텍스트
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(**inputs, max_length=1024)
        result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return result

def process_figure_regions(figure_regions):
    """
    도표 영역 리스트에 대해 Pix2Struct를 이용하여 정보를 추출하고 결과 리스트를 반환합니다.
    
    :param figure_regions: 도표 영역 리스트
    :return: 도표 영역 처리 결과 리스트
    """
    figure_extractor = FigureExtractor()
    results = []
    for region in figure_regions:
        unique_id = region["unique_id"]
        try:
            figure_text = figure_extractor.extract_figure_info(region["image"])
        except Exception as e:
            print(f"{unique_id}에서 도표 추출 실패: {e}")
            continue
        results.append({
            "data_id": unique_id,
            "page_number": region["page_number"],
            "region_type": "도표",
            "content": figure_text,
            "meta": {
                "bounding_box": region["bounding_box"]
            }
        })
    return results
