# Document AI와 생성형 AI를 활용한 문서 데이터 추출 실습

> 처음 배우는 사람을 위한 8시간 Google Colab 실습 과정

![합성 영수증이 OCR과 문서 VLM 경로를 거쳐 검증과 사람 확인으로 이어지는 과정](lessons/assets/course_cover_v2.png)

이 과정은 OCR 라이브러리를 많이 비교하지 않는다. 1교시는 연락처·거래 식별정보를 가린 공개 한국 실물 영수증으로 OCR·VLM·Document AI를 구분하고, 이후 교시는 개인정보 없는 합성 영수증으로 자동화 흐름을 완성한다.

```text
자동화 대상·필드 정의 → OCR/VLM 선택 → 문서 구조화 → JSON → Gradio → 검증 → CSV → 사람 검토
```

## 교육 대상

- Python 변수·함수·딕셔너리를 본 적이 있는 입문자
- OCR, Document AI, Gradio 경험이 없는 학습자
- 반복 문서 업무를 작은 자동화로 시작하고 싶은 실무자

## 수업 원칙

- 교시마다 새 핵심 개념은 최대 3개다.
- 기본 실습은 API 키와 OCR 모델 다운로드가 필요 없다.
- 실제 PaddleOCR와 PaddleOCR-VL은 다운로드가 필요한 선택 실습이다.
- mock 결과에는 항상 `MOCK`이라고 표시한다.
- 실제 개인정보·회사 문서·API 키를 저장소나 공개 Gradio 주소에 올리지 않는다.

## 전체 커리큘럼

| 교시 | 주제 | 교육 목표 | 주요 내용 | 실습 내용 | 산출물 |
| --- | --- | --- | --- | --- | --- |
| **1교시** | 한국 영수증으로 구분하는 OCR·VLM·Document AI | 세 기술의 정의와 처리 과정을 구분한다. | OCR 문자 인식, VLM 관계 해석, Document AI 검증·사람 확인·저장 | 연락처·거래 식별정보를 가린 한국 실물 영수증으로 비교한다. | 기술 비교 JSON |
| **2교시** | OCR 기반 텍스트 추출 실습 | 이미지·PDF 문서에서 텍스트를 추출하는 기본 과정을 이해한다. | OCR 개념, PaddleOCR 3.7과 PP-OCRv5 Korean 선택 경로, 이미지·PDF 입력 처리, 텍스트·위치·신뢰도 확인 | 샘플 이미지 문서에서 준비된 OCR 결과를 확인하고, 선택 실습으로 PaddleOCR를 실행한다. | OCR 추출 결과 텍스트 |
| **3교시** | 문서 구조 이해 및 추출 결과 정제 | OCR 결과의 한계를 이해하고 문서 구조를 고려한 정제 방법을 학습한다. | 문서 레이아웃, 표, 키-값 구조, 항목 관계, OCR 오류 유형, 표 형식 정리 | OCR 결과를 표 또는 항목 단위로 정리한다. | 정제된 문서 데이터 |
| **4교시** | 멀티모달·생성형 AI 기반 핵심 정보 추출 | 문서 VLM과 생성형 AI를 활용해 필요한 정보를 구조화한다. | PaddleOCR-VL 1.6, 프롬프트와 추출 필드 정의, JSON 구조화, 날짜·금액·품목 추출, 근거 없는 값 생성 방지 | 준비된 OCR·VLM 결과를 기반으로 핵심 정보를 JSON으로 변환한다. 실제 모델 호출은 선택 경로로 실행한다. | 구조화된 JSON 데이터 |
| **5교시** | 문서 자동화 웹 애플리케이션 기본 구현 | 문서 처리 기능을 사용자 화면으로 구현하는 방법을 학습한다. | Gradio 개요, 파일 업로드 UI, 처리 결과 화면 구성, Codex 기반 개발 흐름 | Codex와 함께 파일 업로드와 결과 출력 화면을 구현한다. | 기본 문서 업로드 앱 |
| **6교시** | OCR 및 정보 추출 기능 연동 | 업로드한 문서에서 텍스트 추출과 정보 구조화를 한 흐름으로 연결한다. | 업로드 파일 처리, OCR·VLM 선택, 실행 상태와 오류 표시, JSON 결과 출력 | 업로드 → OCR/VLM 선택 → JSON 추출 흐름을 구현한다. | Document AI 미니 애플리케이션 |
| **7교시** | 추출 결과 검증 및 데이터 저장 | AI 추출 결과를 실무에서 활용할 수 있도록 검증하고 저장한다. | 필수값 누락, 날짜·금액 형식, 품목 합계 검증, 오류 표시, 표 변환, 안전한 CSV 다운로드 | 추출 결과 검증 로직을 추가하고 CSV 파일을 생성한다. | 검증 가능한 문서 추출 결과와 CSV |
| **8교시** | 실무 적용 시나리오 설계 및 최종 정리 | 개인 또는 조직 업무에 적용 가능한 자동화 시나리오를 설계한다. | 실패 사례, 개인정보·보안, 사람 검토 절차, 보존·삭제 기준, 업무 적용 아이디어 | 개인별 문서 자동화 적용 시나리오와 개선 방향을 한 장의 카드로 정리한다. | 업무 적용 시나리오 및 개선 과제 |

`EasyOCR`, `Streamlit`, 모델 학습, 데이터베이스, 운영 배포는 2026 필수 과정에 포함하지 않는다. 기본 실습은 준비된 OCR 텍스트와 mock JSON만으로 완료할 수 있고, PaddleOCR와 PaddleOCR-VL 실제 실행은 선택 실습이다.

## 교재와 Colab 바로가기

| 교시 | 교재 | Colab | 산출물 |
| --- | --- | --- | --- |
| 1 | [한국 영수증으로 구분하는 OCR·VLM·Document AI](lessons/01_document_ai_overview.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/01_document_ai_overview.ipynb) | `technology_comparison.json` |
| 2 | [OCR 결과를 눈으로 확인하기](lessons/02_ocr_basic.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/02_ocr_basic.ipynb) | `ocr_text.txt` |
| 3 | [OCR 초안을 정돈된 데이터로](lessons/03_document_structure.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/03_document_structure.ipynb) | `clean_receipt.json` |
| 4 | [PaddleOCR-VL로 문서 구조 읽기](lessons/04_genai_extraction.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/04_genai_extraction.ipynb) | `receipt.json` |
| 5 | [Python 함수에 화면 붙이기](lessons/05_gradio_basic.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/05_gradio_basic.ipynb) | `app_05.py` |
| 6 | [작은 함수들을 한 줄로 연결하기](lessons/06_ocr_ai_integration.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/06_ocr_ai_integration.ipynb) | `app_06.py` |
| 7 | [틀린 값을 걸러 CSV로](lessons/07_validation_export.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/07_validation_export.ipynb) | `receipt.csv` |
| 8 | [자동화의 마지막은 사람 확인](lessons/08_business_application.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/08_business_application.ipynb) | `business_application_card.md` |

현재 Colab 링크는 `document_ai_lecture_2026` 브랜치를 가리킨다. 원격 저장소에 이 브랜치가 올라간 뒤 바로 열린다.

## 가장 쉬운 시작 방법

1. 위 표에서 1교시 Colab 링크를 연다.
2. **런타임 → 모두 실행**을 선택한다.
3. `course_outputs/`에 `technology_comparison.json`이 생겼는지 확인한다.
4. 다음 교시 노트북을 새 탭에서 연다.

각 노트북은 독립 실행형이다. 1교시에는 식별정보를 가린 한국 실물 이미지가, 이후 교시에는 합성 데이터가 내장돼 있다.

## 로컬에서 검증하기

Python 3.12 환경을 준비한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

앱을 실행한다.

```bash
python app.py
```

테스트와 Colab mock 경로를 검증한다.

```bash
python -m compileall .
pytest
python tools/validate_colab_notebooks.py
```

## OCR 또는 설치가 실패했을 때

- 2교시: 내장 `MOCK_OCR_RESULT` 사용
- 3~4교시: 내장 OCR 텍스트와 mock VLM Markdown 사용
- 5교시: Gradio 대신 처리 함수를 셀에서 직접 호출
- 6교시: 오류를 확인한 뒤 `샘플로 계속` 선택
- 7교시: 내장 JSON을 검증하고 Colab 파일 영역에서 CSV 다운로드

자세한 대응은 [문제 해결 가이드](docs/troubleshooting.md)를 참고한다.

## 저장소 구조

```text
.
├── colab/             # 독립 실행형 8교시 노트북
├── lessons/           # Notion 등록용 교재 Markdown과 이미지
├── sample_docs/       # 합성 영수증과 식별정보를 가린 공개 실물 자료
├── sample_outputs/    # OCR·VLM·JSON·CSV mock 결과
├── src/               # 교재와 앱이 공유하는 작은 함수
├── prompts/           # Codex 실습 프롬프트
├── tests/             # 공통 함수와 앱 테스트
├── docs/              # 커리큘럼·환경·강사 운영 문서
├── legacy_materials/  # 과거 자료 보존·재사용 검토
└── app.py             # 최종 Gradio 미니 앱
```

## 운영 문서

- [전체 커리큘럼](docs/curriculum.md)
- [Colab·로컬 환경](docs/environment.md)
- [문제 해결](docs/troubleshooting.md)
- [강사용 운영 가이드](docs/instructor_guide.md)
- [공개 영수증 자료 검토표](docs/public_receipt_datasets.md)
- [최종 검증 보고서](docs/verification_report.md)
- [2026 개편 상태](docs/rebuild_status.md)

## 생성형 AI와 Codex 사용 원칙

- 생성 코드를 그대로 믿지 않고 직접 실행한다.
- 프롬프트에는 목표·맥락·제약조건·완료 기준을 쓴다.
- 원문에 없는 값은 `null`로 처리한다.
- API 키와 실제 개인정보는 프롬프트·노트북·Git에 기록하지 않는다.
- 실제 외부 서비스 연동은 이 입문 과정에 포함하지 않는다. 조직 승인과 데이터 처리 조건을 확인한 별도 환경에서 진행한다.

## 과거 자료

과거 OCR 엔진 비교 노트북과 패키지는 `legacy_materials/source_repo_2025/`에 보존한다. 최신 실행 코드로 사용하지 않는다. 2026 과정의 실행 기준은 `colab/`, `lessons/`, `src/`다.
