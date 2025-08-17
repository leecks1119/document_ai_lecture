# Document AI 강의안 실습 코드 추가 완료

## 개요

깃허브에서 바로 복사해서 사용할 수 있는 완전한 실습 코드를 모든 주요 슬라이드에 추가했습니다. 각 코드는 실무에서 바로 활용 가능하도록 완전한 형태로 구성되었습니다.

## 추가된 실습 코드 목록

### 1. OCR 엔진 성능 비교 실습 (slide_18)
- **파일**: `slide_18_OCR엔진성능비교실습.md`
- **내용**: 
  - Tesseract, EasyOCR, PaddleOCR, Google Vision API 통합 비교
  - 정확도, 속도, 비용 종합 평가 시스템
  - 자동 결과 분석 및 시각화
- **특징**: 
  - 완전한 벤치마크 클래스 (`OCRBenchmark`)
  - 실시간 성능 측정
  - JSON/HTML 리포트 자동 생성

### 2. 전처리 파이프라인 구축 실습 (slide_30)
- **파일**: `slide_30_실습_전처리파이프라인구축.md`
- **내용**:
  - 적응적 문서 품질 평가 시스템
  - 그림자 제거, 슈퍼 해상도, 기울기 보정
  - 저조도 텍스트 강화, 고급 노이즈 제거
- **특징**:
  - 완전한 전처리 클래스 (`DocumentPreprocessor`)
  - 품질 기반 적응적 처리
  - 단계별 결과 시각화

### 3. 멀티모달 LLM OCR 실습 (slide_31_4)
- **파일**: `slide_31_4_멀티모달LLM_OCR실습코드.md`
- **내용**:
  - GPT-4V, Claude 3.5 Sonnet 통합 OCR
  - 이미지 전처리 및 최적화
  - 구조화된 데이터 추출
- **특징**:
  - 완전한 멀티모달 OCR 클래스 (`MultimodalOCR`, `ClaudeOCR`)
  - API 비용 제어 및 최적화
  - 전통 OCR과 성능 비교

### 4. NER 사용자 정의 태깅 실습 (slide_11_2)
- **파일**: `slide_11_2_NER과정보추출.md`
- **내용**:
  - 규칙 기반, spaCy, Transformers, LLM 기반 NER 통합
  - 사용자 정의 도메인별 스키마 구축
  - 앙상블 NER 시스템
- **특징**:
  - 통합 NER 시스템 클래스 (`UnifiedNERSystem`)
  - 도메인별 특화 처리 (`CustomDomainNER`)
  - 성능 평가 및 오류 분석

### 5. 하이브리드 Document AI 시스템 실습 (slide_36)
- **파일**: `slide_36_하이브리드DocumentAI시스템실습.md`
- **내용**:
  - 전통 OCR + 멀티모달 LLM 통합 시스템
  - 적응적 처리 전략 자동 선택
  - 품질 기반 라우팅 및 비용 최적화
- **특징**:
  - 완전한 하이브리드 시스템 (`HybridDocumentAI`)
  - 문서 품질 평가 및 전략 추천
  - 실시간 성능 통계 및 모니터링

### 6. 성능 평가 및 벤치마크 실습 (slide_23)
- **파일**: `slide_23_OCR정확도측정및평가방법론.md`
- **내용**:
  - Character/Word Accuracy, Edit Distance, WER 계산
  - 오류 패턴 분석 및 혼동 행렬
  - 인터랙티브 성능 리포트 생성
- **특징**:
  - 종합 평가 시스템 (`OCRBenchmarkSuite`)
  - Plotly 기반 인터랙티브 차트
  - HTML 리포트 자동 생성

## 실습 코드 특징

### 1. 완전성 (Completeness)
- 모든 import 문과 의존성 명시
- 에러 처리 및 예외 상황 대응
- 실행 가능한 완전한 예제 코드

### 2. 실무 적용성 (Production Ready)
- 설정 파일 기반 구성
- 로깅 및 모니터링 기능
- 성능 최적화 및 비용 제어

### 3. 확장성 (Extensibility)
- 모듈화된 클래스 구조
- 쉬운 커스터마이징 가능
- 새로운 엔진/모델 추가 용이

### 4. 사용자 친화성 (User-Friendly)
- 상세한 주석과 문서화
- 단계별 실행 가이드
- 시각화 및 리포트 자동 생성

## 환경 설정 요구사항

### 기본 패키지
```bash
pip install opencv-python paddlepaddle paddleocr pytesseract
pip install pillow numpy pandas matplotlib seaborn
pip install transformers torch datasets spacy
pip install openai anthropic  # LLM APIs
pip install plotly dash editdistance python-Levenshtein
pip install pydantic fastapi uvicorn  # 구조화 및 API
```

### 추가 설정
```bash
# spaCy 한국어 모델
python -m spacy download ko_core_news_sm

# Tesseract 한국어 패키지 (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

### API 키 설정
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "ANTHROPIC_API_KEY=your_claude_key_here" >> .env
```

## 실습 진행 순서

### 1단계: 기본 환경 구성
- 패키지 설치 및 API 키 설정
- 테스트 이미지 준비

### 2단계: 개별 기술 실습
- OCR 엔진 비교 (slide_18)
- 전처리 파이프라인 (slide_30)
- NER 시스템 (slide_11_2)

### 3단계: 고급 기술 실습
- 멀티모달 LLM OCR (slide_31_4)
- 성능 평가 시스템 (slide_23)

### 4단계: 통합 시스템 구축
- 하이브리드 Document AI (slide_36)
- 종합 성능 평가 및 최적화

## 예상 학습 효과

### 수강생 관점
1. **즉시 실행 가능**: 코드를 복사하여 바로 실습 가능
2. **실무 적용**: 실제 프로젝트에서 활용할 수 있는 완전한 코드
3. **비교 분석**: 다양한 기술의 장단점을 직접 확인
4. **점진적 학습**: 기초부터 고급까지 단계적 실습

### 강사 관점
1. **효율적 진행**: 코드 작성 시간 단축으로 설명에 집중
2. **실시간 데모**: 실제 동작하는 시스템으로 라이브 시연
3. **개별 지도**: 수강생별 코드 실행 결과 비교 분석
4. **질의응답**: 구체적인 구현 관련 질문 대응 가능

## 활용 방안

### 강의 중 활용
- 깃허브 페이지를 공유하여 실시간 코드 복사
- 수강생들이 각자 실행하며 결과 비교
- 코드 수정을 통한 파라미터 실험

### 과제 및 프로젝트
- 기본 코드를 베이스로 한 확장 과제
- 실제 데이터를 활용한 프로젝트 수행
- 성능 개선 및 최적화 실습

### 사후 학습
- 강의 후 복습 및 심화 학습 자료
- 실무 프로젝트 적용 시 참고 코드
- 최신 기술 업데이트 시 코드 확장

## 총 슬라이드 수: 47개

기존 45개에서 2개가 추가되어 총 47개 슬라이드로 구성되었으며, 각 실습 슬라이드는 깃허브에서 바로 복사 가능한 완전한 코드를 포함하고 있습니다.
