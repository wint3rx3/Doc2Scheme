import torch
import easyocr
 
class TextExtractor:
    def __init__(self):
        # EasyOCR 리더 초기화 (한국어, 영어 지원)
        self.reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
    
    def extract_text(self, image):
        """
        이미지에서 텍스트를 추출합니다.
        
        :param image: 크롭된 이미지 영역 (numpy array)
        :return: 추출된 텍스트 (문자열)
        """
        text_result = self.reader.readtext(image, detail=0)
        text = " ".join(text_result).strip()
        return " ".join(text.split())

def process_plain_text_regions(plain_text_regions):
    """
    평문 영역에 대해 OCR을 수행하고 결과를 반환합니다.
    
    :param plain_text_regions: 평문 영역 리스트
    :return: 영역별 결과 딕셔너리 리스트
    """
    extractor = TextExtractor()
    results = []
    for region in plain_text_regions:
        unique_id = region["unique_id"]
        text = extractor.extract_text(region["image"])
        results.append({
            "data_id": unique_id,
            "page_number": region["page_number"],
            "region_type": "평문",
            "content": text,
            "meta": {
                "bounding_box": region["bounding_box"]
            }
        })
    return results
