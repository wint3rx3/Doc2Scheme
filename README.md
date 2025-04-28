# Jokbo_maker
### ⚙️ 모델 파이프 라인
![image](https://github.com/user-attachments/assets/eefb3479-a183-4d6d-8c98-6f7d723750c6)

<br>

### 📂 파일 디렉토리 구조
```
Doc2Scheme/
├── README.md
├── requirements.txt
├── main.py
└── extraction/
    ├── __init__.py
    ├── pdf_processing.py      # PDF → 이미지 변환 및 영역 검출 관련 함수
    ├── yolo_detection.py      # DocLayout-YOLO 모델 로드 및 영역 검출 함수
    ├── ocr_extraction.py      # 평문 OCR 텍스트 추출
    ├── table_extraction.py    # 표 구조 인식 및 텍스트 추출
    └── figure_extraction.py   # 도표 영역 텍스트 추출
```
