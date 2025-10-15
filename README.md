# Document AI 강의 자료 📚

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/master/notebooks/Lab01_개발환경구축.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**실습 중심 Document AI 완전 정복**

OCR, 이미지 전처리, NER까지 실전 Document AI 기술을 Google Colab에서 바로 실습합니다.

---

## 🚀 바로 시작하기

### 1️⃣ Colab에서 노트북 열기 (가장 쉬움!)

각 실습 노트북 상단의 **"Open in Colab"** 배지를 클릭하면 바로 실습 시작!

**또는 직접 URL로 접속:**
```
https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/master/notebooks/Lab01_개발환경구축.ipynb
```

### 2️⃣ 패키지 설치 (노트북 첫 셀)

모든 실습 코드가 포함된 패키지를 한 줄로 설치:

```python
!pip install -q git+https://github.com/leecks1119/document_ai_lecture.git
!apt-get install -y tesseract-ocr tesseract-ocr-kor
```

### 3️⃣ 바로 사용!

```python
from docai_course import OCRBenchmark, DocumentPreprocessor, UnifiedNERSystem

# OCR 비교
benchmark = OCRBenchmark()
results = benchmark.run_comparison('image.jpg', 'ground_truth')

# 이미지 전처리
preprocessor = DocumentPreprocessor()
processed, binary, metrics = preprocessor.adaptive_preprocessing_pipeline('image.jpg')

# 정보 추출
ner = UnifiedNERSystem()
entities = ner.rule_based_ner(text)
```

---

## 📚 실습 노트북 (10개)

| Lab | 제목 | 난이도 | 시간 |
|-----|------|--------|------|
| **01** | [개발환경 구축](notebooks/Lab01_개발환경구축.ipynb) | ⭐ | 10분 |
| **03** | [PaddleOCR 기본](notebooks/Lab03_PaddleOCR.ipynb) | ⭐⭐ | 20분 |
| **04** | [OCR 엔진 비교](notebooks/Lab04_OCR엔진비교.ipynb) | ⭐⭐⭐ | 30분 |
| **05** | [신뢰도 측정](notebooks/Lab05_신뢰도측정.ipynb) | ⭐⭐⭐ | 25분 |
| **06** | [이미지 전처리](notebooks/Lab06_이미지전처리.ipynb) | ⭐⭐⭐⭐ | 40분 |
| **07** | [OCR 앙상블](notebooks/Lab07_앙상블.ipynb) | ⭐⭐⭐⭐ | 35분 |
| **08** | [표 검출](notebooks/Lab08_표검출.ipynb) | ⭐⭐⭐⭐ | 40분 |
| **09** | [NER 정보추출](notebooks/Lab09_NER정보추출.ipynb) | ⭐⭐⭐ | 30분 |
| **10** | [토이 프로젝트](notebooks/Lab10_토이프로젝트.ipynb) | ⭐⭐⭐⭐⭐ | 60분 |
| **11** | [전체 테스트](notebooks/Lab11_전체테스트.ipynb) | ⭐⭐⭐⭐ | 40분 |

---

## 📦 패키지 구조

```
document_ai_lecture/
├── notebooks/                 # 실습 노트북 (10개)
│   ├── Lab01_개발환경구축.ipynb
│   ├── Lab03_PaddleOCR.ipynb
│   ├── ... (Lab04~10)
│   └── Lab11_전체테스트.ipynb
│
├── docai_course/             # Python 패키지
│   ├── ocr/                  # OCR 엔진
│   ├── preprocessing/        # 전처리
│   ├── ner/                  # NER
│   └── hybrid/               # 하이브리드
│
├── setup.py
└── requirements.txt
```

---


## 📊 기술 스택

- **OCR**: Tesseract, PaddleOCR, EasyOCR
- **전처리**: OpenCV, PIL, NumPy
- **NER**: Regex, spaCy
- **Python**: 3.8+ / PyTorch / Pandas

---

## 🔗 링크
- **Notion**: [강의자료](https://vivid-mailbox-751.notion.site/Document-AI-281707c7ae7581beb748feca63ac4e16)
- **GitHub**: https://github.com/leecks1119/document_ai_lecture
- **Colab 시작**: [Lab01 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/master/notebooks/Lab01_개발환경구축.ipynb)

---

**Happy Learning! 🚀**