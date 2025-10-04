# Document AI 강의 자료

**실습 중심 Document AI 완전 정복 과정**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

이 저장소는 8시간 분량의 Document AI 실습 강의 자료를 포함하고 있습니다.

## 📦 패키지 설치 (Google Colab)

### 방법 1: GitHub에서 직접 설치 (권장 ⭐)
```python
# Google Colab 셀에서 실행
!pip install git+https://github.com/leecks1119/document_ai_lecture.git
```

### 방법 2: 클론 후 설치
```python
# 저장소 클론
!git clone https://github.com/leecks1119/document_ai_lecture.git
!cd document_ai_lecture && pip install -e .
```

## 🚀 빠른 시작

### Google Colab에서 사용
```python
# 1. 패키지 설치
!pip install git+https://github.com/leecks1119/document_ai_lecture.git

# 2. 임포트
from docai_course import OCRBenchmark, DocumentPreprocessor

# 3. OCR 벤치마크 실행
benchmark = OCRBenchmark()
results = benchmark.run_comparison(
    image_path='sample.jpg',
    ground_truth='정답 텍스트'
)

# 4. 결과 저장
benchmark.save_results(results)
```

## 📚 주요 클래스

### 1. OCRBenchmark
여러 OCR 엔진의 성능을 비교합니다.
```python
from docai_course.ocr import OCRBenchmark

benchmark = OCRBenchmark()
results = benchmark.run_comparison('image.jpg', 'ground_truth')
```

### 2. DocumentPreprocessor  
이미지 전처리를 수행합니다.
```python
from docai_course.preprocessing import DocumentPreprocessor

preprocessor = DocumentPreprocessor()
processed, binary, metrics = preprocessor.adaptive_preprocessing_pipeline('image.jpg')
```

### 3. UnifiedNERSystem
통합 NER 시스템입니다.
```python
from docai_course.ner import UnifiedNERSystem

ner = UnifiedNERSystem()
entities = ner.ensemble_ner(text)
```

### 4. HybridDocumentAI
하이브리드 Document AI 시스템입니다.
```python
from docai_course.hybrid import HybridDocumentAI

system = HybridDocumentAI()
result = system.process_document('image.jpg', schema)
```

## 📂 프로젝트 구조

```
document_ai_lecture/
├── docai_course/              # 메인 패키지
│   ├── __init__.py
│   ├── ocr/                   # OCR 엔진들
│   │   ├── __init__.py
│   │   └── benchmark.py       # OCRBenchmark 클래스
│   ├── preprocessing/         # 전처리
│   │   ├── __init__.py
│   │   └── preprocessor.py    # DocumentPreprocessor 클래스
│   ├── ner/                   # NER 시스템
│   │   ├── __init__.py
│   │   └── unified_ner.py     # UnifiedNERSystem 클래스
│   └── hybrid/                # 하이브리드 시스템
│       ├── __init__.py
│       └── system.py          # HybridDocumentAI 클래스
├── slide_*.md                 # 강의 슬라이드 (44개)
├── setup.py                   # 패키지 설정
├── requirements.txt           # 의존성
└── README.md                  # 이 파일

```

## 🎓 강의 커리큘럼

### 1부: Document AI 기초 (2시간)
- Document AI 정의 및 비즈니스 가치
- 시장 동향 및 도입 필요성
- 작동 원리 및 기술 스택

### 2부: OCR 기술 심화 (2시간)
- OCR 엔진 비교 실습 (Tesseract, PaddleOCR, EasyOCR)
- 한글 OCR 최적화
- 성능 측정 및 평가

### 3부: 이미지 전처리 (2시간)
- 노이즈 제거, 이진화, 기울기 보정
- 품질 평가 자동화
- 전처리 파이프라인 구축

### 4부: LLM 통합 (2시간)
- LLM 기반 OCR 보정
- 하이브리드 시스템 구축
- Cursor AI 활용 토이 프로젝트

## 💡 실습 예제

각 슬라이드 파일(`slide_*.md`)에는 상세한 실습 코드가 포함되어 있습니다.

## 🤝 기여

이슈와 PR은 언제나 환영합니다!

## 📄 라이선스

MIT License

## 👨‍🏫 강사 소개

**이찬희 프로**
- 삼성 SDS 보안 PM · AI-SOC 솔루션 업무 리딩
- Gen AI 및 보안 분석 전문가

## 📞 문의

강의 관련 문의는 GitHub Issues를 이용해주세요.

---

**Happy Learning! 🚀**

