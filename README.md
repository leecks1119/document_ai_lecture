# Document AI와 생성형 AI를 활용한 문서 데이터 추출 실습

> 처음 배우는 사람을 위한 8시간 Google Colab 실습 과정

![학습자가 합성 영수증을 OCR, 구조화, 검증, 표로 처리하는 과정](lessons/assets/course_cover.png)

이 과정은 OCR 라이브러리를 많이 비교하지 않는다. 개인정보가 없는 합성 영수증 한 장을 사용해 다음 흐름을 직접 완성한다.

```text
필드 정의 → OCR 결과 확인 → 텍스트 정제 → JSON → Gradio → 검증 → CSV → 사람 검토
```

## 교육 대상

- Python 변수·함수·딕셔너리를 본 적이 있는 입문자
- OCR, Document AI, Gradio 경험이 없는 학습자
- 반복 문서 업무를 작은 자동화로 시작하고 싶은 실무자

## 수업 원칙

- 교시마다 새 핵심 개념은 최대 3개다.
- 기본 실습은 API 키와 OCR 모델 다운로드가 필요 없다.
- 실제 EasyOCR는 선택 실습이다. 생성형 AI API는 연결 전 준비사항만 확인한다.
- mock 결과에는 항상 `MOCK`이라고 표시한다.
- 실제 개인정보·회사 문서·API 키를 저장소나 공개 Gradio 주소에 올리지 않는다.

## 8교시 바로가기

| 교시 | 교재 | Colab | 산출물 |
| --- | --- | --- | --- |
| 1 | [OCR보다 먼저 정할 것](lessons/01_document_ai_overview.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/01_document_ai_overview.ipynb) | `field_spec.json` |
| 2 | [OCR 결과를 눈으로 확인하기](lessons/02_ocr_basic.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/02_ocr_basic.ipynb) | `ocr_text.txt` |
| 3 | [OCR 초안을 정돈된 데이터로](lessons/03_document_structure.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/03_document_structure.ipynb) | `clean_receipt.json` |
| 4 | [필요한 값만 JSON으로](lessons/04_genai_extraction.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/04_genai_extraction.ipynb) | `receipt.json` |
| 5 | [Python 함수에 화면 붙이기](lessons/05_gradio_basic.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/05_gradio_basic.ipynb) | `app_05.py` |
| 6 | [작은 함수들을 한 줄로 연결하기](lessons/06_ocr_ai_integration.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/06_ocr_ai_integration.ipynb) | `app_06.py` |
| 7 | [틀린 값을 걸러 CSV로](lessons/07_validation_export.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/07_validation_export.ipynb) | `receipt.csv` |
| 8 | [자동화의 마지막은 사람 확인](lessons/08_business_application.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/08_business_application.ipynb) | `business_application_card.md` |

현재 Colab 링크는 `document_ai_lecture_2026` 브랜치를 가리킨다. 원격 저장소에 이 브랜치가 올라간 뒤 바로 열린다.

## 가장 쉬운 시작 방법

1. 위 표에서 1교시 Colab 링크를 연다.
2. **런타임 → 모두 실행**을 선택한다.
3. `course_outputs/`에 `field_spec.json`이 생겼는지 확인한다.
4. 다음 교시 노트북을 새 탭에서 연다.

각 노트북은 독립 실행형이다. 이전 교시 파일을 잃어도 내장 합성 데이터로 계속할 수 있다.

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
- 3~4교시: 내장 OCR 텍스트와 mock JSON 사용
- 5교시: Gradio 대신 처리 함수를 셀에서 직접 호출
- 6교시: 오류를 확인한 뒤 `샘플로 계속` 선택
- 7교시: 내장 JSON을 검증하고 Colab 파일 영역에서 CSV 다운로드

자세한 대응은 [문제 해결 가이드](docs/troubleshooting.md)를 참고한다.

## 저장소 구조

```text
.
├── colab/             # 독립 실행형 8교시 노트북
├── lessons/           # Notion 등록용 교재 Markdown과 이미지
├── sample_docs/       # 개인정보 없는 합성 영수증
├── sample_outputs/    # OCR·JSON·CSV mock 결과
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
- [최종 검증 보고서](docs/verification_report.md)
- [2026 개편 상태](docs/rebuild_status.md)

## 생성형 AI와 Codex 사용 원칙

- 생성 코드를 그대로 믿지 않고 직접 실행한다.
- 프롬프트에는 목표·맥락·제약조건·완료 기준을 쓴다.
- 원문에 없는 값은 `null`로 처리한다.
- API 키와 실제 개인정보는 프롬프트·노트북·Git에 기록하지 않는다.
- 실제 API 호출은 이 입문 과정에 포함하지 않는다. 조직 승인과 데이터 처리 조건을 확인한 별도 환경에서 진행한다.

## 과거 노트북

루트 `notebooks/`와 `docai_course/`는 과거 OCR 중심 과정의 참고 자료다. 2026 과정의 실행 기준은 `colab/`, `lessons/`, `src/`다.
