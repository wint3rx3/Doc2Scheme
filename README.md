# Doc2Scheme

이 프로젝트는 PDF 문서 내 페이지 레이아웃을 추출하여 평문, 표, 도표 등 각 영역을 검출하고, 해당 영역에서 텍스트 및 정보를 자동으로 추출하는 파이프라인입니다. 주요 기술 및 도구로는 PDF → 이미지 변환, DocLayout-YOLO를 이용한 영역 검출, EasyOCR, 셀 기반 표 추출, 그리고 Pix2Struct 기반 도표 설명 생성 등이 있습니다.

## 파이프라인
![image](https://github.com/user-attachments/assets/fa1ac4f6-0b0d-4e85-9de1-169b77b5fcb4)

## 팀원 소개
- 김소현 : 레이아웃 검출
- 김영홍 : 도표 영역 텍스트 추출
- 우동협 : 표 구조 인식 및 텍스트 추출
- 김윤영 : 표 구조 인식 및 텍스트 추출
- 신진섭 : LLM을 이용한 요약문, 설명문 생성

## 프로젝트 구조
```plaintext
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

## 모델 파이프라인 구성 설명

- **pdf_processing**  
  Poppler를 이용하여 PDF 파일을 고해상도 이미지(각 페이지별)로 변환합니다.
  - 포플러(Poppler)는 PDF 문서 렌더링을 수행하는 라이브러리이다. 현재 그놈과 KDE의 PDF 뷰어에서 사용되고 있으며 Freedesktop.org에서 호스팅되고 있다. (출처 : 위키피디아)
  - PDF를 이미지화 하는 이유는? : PDF는 기본적으로 벡터 형식이기 때문에, 문서 내 요소들을 분석하기 위해서는 이를 픽셀 기반의 이미지로 변환해야 합니다. 이미지 형태는 딥러닝 기반의 객체 검출, OCR 등 다양한 컴퓨터 비전 모델에 바로 적용할 수 있습니다.
  
- **yolo_detection**  
  [DocLayout-YOLO (YOLOv10 기반 사전학습 모델)](https://github.com/opendatalab/DocLayout-YOLO)을 활용하여 각 페이지에서 영역(평문, 표, 도표 등)을 검출하고, 중복된 박스를 IoU 기준으로 제거합니다.
  - 문서 내의 텍스트 블록, 표, 도표 등 문서 구성 요소를 대상으로 학습되어 문서의 고유한 구조적 패턴을 인식하는 데 최적화되어 있음
  - YOLO 계열 모델의 장점을 그대로 계승하여 빠른 추론 속도와 상대적으로 가벼운 모델 구조를 유지합니다. 이는 대용량 문서나 여러 페이지를 실시간으로 처리할 때 유리합니다.
  - IOU 후처리 계산을 통해 곂치는 부분을 탐지하고 가장 신뢰도가 높은 구역을 탐지하여 바운딩 박스를 도출합니다.

- **ocr_extraction**  
  [EasyOCR](https://github.com/JaidedAI/EasyOCR)을 사용하여 평문 영역에서 텍스트를 추출합니다.
  - EasyOCR은 문자 영역 인식(Detection) + 문자 인식(Recognition) 기능을 모두 하는 프레임워크이다. 2020년에 나타난 비교적 최신 OCR로 현재까지 많은 사람들이 이용하고 있고 80가지가 넘는 언어를 지원한다. 현재까지도 활발히 업데이트가 이루어지고 있다.
  - Detection은 Clova AI의 CRAFT를, Recognition은 CRNN을, 특징 추출을 위해 ResNet을, Sequence labeling을 위해 LSTM을, 그리고 decoder로 CTC를 사용한다. 또한 Recognition의 training pipeline으로 Clova AI의 deep-text-recognition-benchmark를 사용한다.
  - 상대적으로 가볍고 빠른 모델을 사용하여, CPU 환경에서도 실행 가능하다.
  
- **table_extraction**  
  OpenCV의 Canny Edge Detection, HoughLinesP, EasyOCR을 활용해 표의 셀과 그리드 정보를 추출하여 표 내 텍스트 및 구조 정보를 제공합니다.
  - 에지 검출 (Canny Edge Detection) : 이미지 내에서 급격한 밝기 변화(즉, 에지)를 검출하여, 선의 후보 영역을 식별합니다.
  - 경계선 검출 (HoughLinesP) : Canny로 검출된 에지 이미지에 대해 Hough 변환(HoughLinesP)을 적용하여, 직선 형태의 선들을 검출합니다. 이 단계에서는 선의 시작점, 끝점, 길이 및 방향 등의 정보를 얻습니다.
  - 수직-수평선 판단, 중복 제거 및 교차점 검출, 테두리 추가 등을 수행합니다. 그리고 각 셀마다 EasyOCR을 통해 텍스트를 추출합니다.

- **figure_extraction**  
  [deplot_kr (Pix2Struct 기반 사전학습 모델)](https://huggingface.co/brainventures/deplot_kr) 사용하여 도표 영역에 대한 설명 텍스트를 생성합니다.
  - deplot_kr은 google의 pix2struct 구조를 기반으로 한 한국어 image-to-data 모델입니다. DePlot 모델을 한국어 차트 이미지-텍스트 쌍 데이터세트(30만 개)를 이용하여 fine-tuning 했습니다. (출처 : brainuniverse 허깅페이스)
  - 현재 프로젝트에서는 LLM을 이용한 요약문, 설명문 출력이 필요 없으니, 단순 결과 값만 활용함
  
- **최종 결과 출력**  
  추출된 모든 영역의 정보는 페이지 번호, 영역 타입, 콘텐츠, 그리고 메타데이터(바운딩 박스 등)를 포함하여 JSON 및 CSV (직관적인 결과 확인용) 파일로 저장됩니다.


