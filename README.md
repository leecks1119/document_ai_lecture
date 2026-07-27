# Document AI와 생성형 AI를 활용한 문서 데이터 추출 실습

> Python을 한 번 사용해 본 일반 실무자가 실제 영수증 한 장으로 작은 Document AI 프로토타입을 완성하는 하루 8교시 과정

![영수증 한 장이 판독·구조화·검증·사람 확인을 거쳐 업무 데이터가 되는 과정](lessons/assets/course_cover_v2.png)

## 이 강의가 남겨야 할 한 문장

> “영수증으로 해보니 원리를 알겠다. 우리 회사의 견적서나 신청서도 이런 식으로 자동화해볼 수 있겠는데?”

학습자는 식별정보를 가린 실제 한국 영수증 한 장을 판독하고, 필요한 값을 JSON으로 구조화하고, 원문과 대조해 검증한 뒤 `receipt_result.xlsx`로 내려받는다. 마지막에는 같은 방식을 견적서·신청서·거래명세서에 적용할 수 있는지 검토한다. 발주서는 이번 과정에서 제외한다.

과정을 마치면 다음을 할 수 있다.

- 실제 문서 한 장으로 작동하는 작은 프로토타입을 직접 만들 수 있다.
- OCR·Multimodal AI·VLM·Document AI·IDP의 역할과 한계를 설명할 수 있다.
- 자동화할 필드, 검증 규칙, 사람 확인, 오류 영향을 정할 수 있다.
- 우리 업무에서 작게 시험할 PoC 후보를 검토할 수 있다.

## 과정 기준

| 항목 | 내용 |
| --- | --- |
| 기준일 | 2026-07-27 |
| 교육 시간 | 하루 8시간, 8교시 연속 |
| 교육 대상 | Python을 한 번 사용해 봤지만 문법을 모두 알지는 못하는 일반 실무자 |
| 교육 인원 | 20명, 주강사 1명, 보조강사 없음 |
| 실습 방식 | 1인 1문제, 한 번에 문서 한 장 |
| 실습 환경 | 모든 필수 실습을 Google Colab에서 수행 |
| 코드 제공 | 시작 코드·빈칸·힌트·전체 정답을 처음부터 공개 |
| 한국어 OCR | PaddleOCR 3.7 + PP-OCRv5 Korean |
| 문서 VLM | PaddleOCR-VL 1.6 또는 상용 VLM의 강사 시연 1회 |
| 웹앱 | Streamlit 1.60.0, Colab에서는 AppTest로 검사 |
| 기본 비용 | 학습자 API 키·별도 API 과금 없음 |
| 최종 결과 | `receipt_result.xlsx`를 내려받는 작은 프로토타입 |

## 먼저 구분할 다섯 용어

업계에서 용어의 경계는 공급자에 따라 일부 겹친다. 이 과정에서는 학습과 구현을 위해 다음 기준을 사용한다.

| 용어 | 이 과정의 정의 | 보장하지 않는 것 |
| --- | --- | --- |
| OCR | 문서 이미지에서 텍스트를 인식하는 기술. 제품에 따라 페이지·좌표·신뢰도도 제공할 수 있다. | 업무 의미, 계산의 정확성, 승인 여부 |
| Multimodal AI | 이미지·텍스트·음성 등 둘 이상의 데이터 형식을 다루는 넓은 범주 | 개별 출력의 사실성 |
| VLM | 이미지와 언어를 함께 다루는 Multimodal AI의 한 종류 | 원문 충실도, 표 관계, JSON 값의 정확성 |
| Document AI | 분류·OCR·레이아웃·필드 추출·정규화 등 문서를 구조화하는 기술·제품 역량 | 조직의 승인·예외·운영 절차 전체 |
| IDP | Document AI 역량을 접수·검증·예외·사람 확인·업무 연결·평가에 결합한 운영 범위 | 검토 없는 완전 자동 정확성 |

OCR과 VLM은 문서와 업무에 따라 선택하거나 조합한다. `OCR → VLM → Document AI`를 반드시 거치는 고정 순서가 아니다.

## 하루 동안 따라갈 전체 지도

```text
0 목표·스키마
  → 1 접수
  → 2 형식 라우팅·분리
  → 3 품질 확인·전처리
  → 4 텍스트·레이아웃 추출
  → 5 문서 유형 분류
  → 6 필드·표 구조화
  → 7 정규화
  → 8 검증
  → 9 AUTO_ACCEPT·REVIEW·REJECT 결정
  → 10 사람 검토
  → 11 Excel·업무 시스템 연결
  → 12 관측·평가·개선
```

보안·개인정보·접근 통제·감사 기록·보존·삭제는 특정 한 단계가 아니라 전체를 가로지른다. 1교시에서 이 지도를 실제 영수증에 대입하고, 2~8교시에서 필요한 부분을 하나씩 구현한다.

## 8교시 커리큘럼

공지된 교시 주제는 변경하지 않는다.

| 교시 | 공지된 주제 | 이번 교시의 한 가지 메시지 | 기본 실습 | 산출물 |
| --- | --- | --- | --- | --- |
| 1 | 한국 영수증으로 구분하는 OCR·VLM·Document AI | 문서 자동화는 모델 하나가 아니라 목표부터 운영 개선까지 이어지는 과정이다. | 용어 관계와 0~12 지도에서 영수증의 원문·근거·검증·처리 결정을 추적한다. | `receipt_pipeline_trace.json` |
| 2 | OCR 기반 텍스트 추출 실습 | OCR 결과는 정답이 아니라 원본과 대조할 판독 결과다. | 비식별 영수증 한 장에 PP-OCRv5 Korean을 실행하고 오류를 표시한다. | `ocr_result.json` |
| 3 | 문서 구조 이해 및 추출 결과 정제 | 읽힌 글자를 키-값과 반복 행으로 재구성해야 업무 데이터가 된다. | OCR 원문을 보존하며 상호명·날짜·품목·합계를 정리한다. | `clean_receipt.json` |
| 4 | 멀티모달·생성형 AI 기반 핵심 정보 추출 | VLM의 구조 초안은 근거와 검증 전에는 확정 데이터가 아니다. | 준비된 VLM 결과에서 근거 있는 값만 JSON으로 옮긴다. | `receipt.json` |
| 5 | 문서 자동화 웹 애플리케이션 기본 구현 | 처리 함수에 입력·실행·결과 화면을 붙이면 사람이 사용할 수 있는 도구가 된다. | Streamlit 코드를 만들고 AppTest로 검사한다. | `app_05.py` |
| 6 | OCR 및 정보 추출 기능 연동 | 입력부터 JSON까지의 각 단계와 실패 위치를 명시적으로 연결한다. | `파일 → OCR/준비 결과 → 추출 → JSON` 경로를 완성한다. | `app_06.py` |
| 7 | 추출 결과 검증 및 데이터 저장 | 규칙과 사람 확인을 통과한 값만 업무용 Excel이 된다. | 오류 저장을 차단하고 정상 결과를 Excel로 내려받는다. | `receipt_result.xlsx` |
| 8 | 실무 적용 시나리오 설계 및 최종 정리 | 같은 파이프라인을 재사용하되 PoC 적합성은 업무별로 판단한다. | 견적서·신청서·거래명세서 중 하나의 필드·규칙·중단 조건을 정한다. | `poc_candidate_card.md` |

## 교재와 Colab

| 교시 | 교재 | Colab | 산출물 |
| --- | --- | --- | --- |
| 1 | [한국 영수증으로 구분하는 OCR·VLM·Document AI](lessons/01_document_ai_overview.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/01_document_ai_overview.ipynb) | `receipt_pipeline_trace.json` |
| 2 | [OCR 기반 텍스트 추출 실습](lessons/02_ocr_basic.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/02_ocr_basic.ipynb) | `ocr_result.json` |
| 3 | [문서 구조 이해 및 추출 결과 정제](lessons/03_document_structure.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/03_document_structure.ipynb) | `clean_receipt.json` |
| 4 | [멀티모달·생성형 AI 기반 핵심 정보 추출](lessons/04_genai_extraction.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/04_genai_extraction.ipynb) | `receipt.json` |
| 5 | [문서 자동화 웹 애플리케이션 기본 구현](lessons/05_streamlit_basic.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/05_streamlit_basic.ipynb) | `app_05.py` |
| 6 | [OCR 및 정보 추출 기능 연동](lessons/06_ocr_ai_integration.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/06_ocr_ai_integration.ipynb) | `app_06.py` |
| 7 | [추출 결과 검증 및 데이터 저장](lessons/07_validation_export.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/07_validation_export.ipynb) | `receipt_result.xlsx` |
| 8 | [실무 적용 시나리오 설계 및 최종 정리](lessons/08_business_application.md) | [열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/08_business_application.ipynb) | `poc_candidate_card.md` |

Colab 링크는 `document_ai_lecture_2026` 브랜치를 가리킨다. 원격 저장소에 이 브랜치가 올라간 뒤 바로 열린다.

## 수업 운영 원칙

- 개념은 전체 약 10%, 실행·비교·수정·복구·검증은 약 90%로 운영한다.
- 1교시는 전체 절차를 안내형 실습으로 이해하고, 2~8교시는 교시당 새 개념을 세 개 이내로 제한한다.
- 실제 OCR은 2교시에서 3분 동안 먼저 시도하고, 해결되지 않으면 준비 결과로 전환한다.
- 준비 결과에는 `MOCK` 또는 `준비 결과`를 분명히 표시한다.
- 학습자는 API 키를 만들거나 비용을 결제하지 않는다.
- 상용 VLM은 강사가 비식별 샘플로 한 번만 시연한다.
- 개인 영수증·회사 문서는 공개 웹앱 주소나 외부 API에 올리지 않는다.
- Streamlit 필수 실습은 공개 터널 없이 Colab의 `streamlit.testing.v1.AppTest`로 검사한다.

개인 영수증을 사용하려면 카드번호·승인번호·현금영수증 번호·전화번호·회원번호 등을 먼저 가리고, 실습 뒤 Colab 출력과 런타임 파일을 삭제한다.

## 가장 쉬운 시작

1. [1교시 교재](lessons/01_document_ai_overview.md)를 열어 용어 관계와 전체 지도를 본다.
2. 1교시 Colab을 열고 위에서 아래로 실행한다.
3. 빈칸이 막히면 같은 노트북의 전체 정답으로 즉시 복구한다.
4. 각 교시가 끝날 때 산출물 한 개를 내려받는다.

각 노트북은 독립 실행형이며 이전 교시 산출물의 완성 복구본을 포함한다.

## 개발자·강사용 로컬 검증

학습자의 필수 경로는 Colab이다. 저장소 전체를 검증하거나 최종 Streamlit 앱을 시연할 때만 로컬 환경을 사용한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```bash
streamlit run app.py
```

```bash
python -m compileall .
pytest
python tools/validate_course_materials.py
python tools/validate_colab_notebooks.py
```

## 저장소 구조

```text
.
├── README.md          # 저장소의 유일한 README
├── colab/             # 독립 실행형 8교시 Colab 노트북
├── lessons/           # 교시별 Notion 등록용 Markdown과 이미지
├── sample_docs/       # 합성·비식별 공개 실물 문서
├── sample_outputs/    # 준비된 OCR·VLM·JSON·Excel 결과
├── src/               # OCR·추출·검증·Excel 공통 함수
├── prompts/           # 선택형 생성 AI 작업 템플릿
├── tests/             # 공통 함수와 Streamlit AppTest
├── docs/              # 커리큘럼·환경·운영·검증 문서
├── legacy_materials/  # 과거 자료 보존과 재사용 검토
└── app.py             # 최종 Streamlit 미니 앱
```

저장소에서 관리하는 README는 루트의 이 파일 하나뿐이다. 가상환경이나 캐시가 자체적으로 포함한 README는 Git 관리 대상이 아니다.

## 운영 문서

- [전체 커리큘럼](docs/curriculum.md)
- [Colab·로컬 환경](docs/environment.md)
- [문제 해결](docs/troubleshooting.md)
- [강사용 운영 가이드](docs/instructor_guide.md)
- [공개 영수증 자료 검토표](docs/public_receipt_datasets.md)
- [최종 검증 보고서](docs/verification_report.md)
- [2026 개편 상태](docs/rebuild_status.md)

## 기술 근거

- [Google Cloud Document AI 개요](https://cloud.google.com/document-ai/docs/overview)
- [AWS Intelligent Document Processing 설명](https://aws.amazon.com/what-is/intelligent-document-processing/)
- [PaddleOCR PP-OCRv5 다국어 모델](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
- [PaddleOCR-VL 1.6](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

기술 버전·지원 범위·비용·보안 정책은 2026-07-27 기준 공식 자료를 우선한다. 2024년 이후 널리 알려진 사례는 문제 구조와 설명 방식에 활용할 수 있지만, 현재 사양의 근거로 그대로 사용하지 않는다.
