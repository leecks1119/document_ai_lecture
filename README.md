# Document AI와 생성형 AI를 활용한 문서 데이터 추출 실습

> Python을 한 번 사용해 본 일반 실무자가 실제 영수증 한 장으로 작은 Document AI 프로토타입을 완성하는 하루 8교시 과정

![영수증 한 장이 판독·구조화·검증·사람 확인을 거쳐 업무 데이터가 되는 과정](lessons/assets/course_cover_v2.png)

## 이 강의가 남겨야 할 한 문장

> “영수증으로 해보니 원리를 알겠다. 우리 회사의 견적서나 신청서도 이런 식으로 자동화해볼 수 있겠는데?”

여러분은 식별정보를 가린 한국 영수증 한 장을 판독하고, 필요한 값을 JSON으로 구조화하고, 원본과 대조해 검증한 뒤 `receipt_result.xlsx`로 내려받습니다. 마지막에는 같은 방식을 견적서·신청서·거래명세서에 적용할 수 있는지 검토합니다. 발주서는 이번 입문 과정에서 다루지 않습니다.

과정을 마치면 다음을 할 수 있습니다.

- OCR·Multimodal AI·VLM·Document AI·IDP의 역할과 한계를 설명합니다.
- 실제 문서 한 장으로 작동하는 작은 프로토타입을 직접 만듭니다.
- 자동화할 필드, 검증 규칙, 사람 확인, 오류 영향을 정합니다.
- 우리 업무에서 작게 시험할 PoC 후보를 검토합니다.

## 하루 동안 만들 결과

```text
식별정보를 가린 영수증 한 장
  → OCR 또는 준비 결과
  → 키-값·반복 행 정리
  → 업무용 JSON
  → 규칙 검증
  → 원본 대조와 사람 승인
  → receipt_result.xlsx
```

모든 필수 실습은 Google Colab에서 문서 한 장씩 진행합니다. 시작 코드·빈칸·힌트·전체 정답이 제공되며 API 키나 별도 결제가 필요하지 않습니다.

## 8교시 커리큘럼

| 교시 | 공지된 주제 | 이번 교시의 한 가지 메시지 | 산출물 |
| --- | --- | --- | --- |
| 1 | 한국 영수증으로 구분하는 OCR·VLM·Document AI | 문서 자동화는 모델 하나가 아니라 목표부터 운영 개선까지 이어지는 과정입니다. | `receipt_pipeline_trace.json` |
| 2 | OCR 기반 텍스트 추출 실습 | OCR 결과는 정답이 아니라 원본과 대조할 판독 결과입니다. | `ocr_result.json` |
| 3 | 문서 구조 이해 및 추출 결과 정제 | 읽힌 글자를 키-값과 반복 행으로 재구성해야 업무 데이터가 됩니다. | `clean_receipt.json` |
| 4 | 멀티모달·생성형 AI 기반 핵심 정보 추출 | VLM의 구조 초안은 근거와 검증 전에는 확정 데이터가 아닙니다. | `receipt.json` |
| 5 | 문서 자동화 웹 애플리케이션 기본 구현 | 처리 함수에 입력·실행·결과 화면을 붙이면 사용할 수 있는 도구가 됩니다. | `app_05.py` |
| 6 | OCR 및 정보 추출 기능 연동 | 파일에서 JSON까지 단계를 연결하면 실패 위치를 찾을 수 있습니다. | `app_06.py` |
| 7 | 추출 결과 검증 및 데이터 저장 | 규칙과 사람 확인을 통과한 값만 업무용 Excel이 됩니다. | `receipt_result.xlsx` |
| 8 | 실무 적용 시나리오 설계 및 최종 정리 | 같은 흐름을 재사용하되 PoC 적합성은 업무별로 판단합니다. | `poc_candidate_card.md` |

## 교재와 Colab

| 교시 | 교재 | Colab |
| --- | --- | --- |
| 1 | [한국 영수증으로 구분하는 OCR·VLM·Document AI](lessons/01_document_ai_overview.md) | [실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/01_document_ai_overview.ipynb) |
| 2 | [OCR 기반 텍스트 추출 실습](lessons/02_ocr_basic.md) | [실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/02_ocr_basic.ipynb) |
| 3 | [문서 구조 이해 및 추출 결과 정제](lessons/03_document_structure.md) | [실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/03_document_structure.ipynb) |
| 4 | [멀티모달·생성형 AI 기반 핵심 정보 추출](lessons/04_genai_extraction.md) | [실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/04_genai_extraction.ipynb) |
| 5 | [문서 자동화 웹 애플리케이션 기본 구현](lessons/05_streamlit_basic.md) | [실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/05_streamlit_basic.ipynb) |
| 6 | [OCR 및 정보 추출 기능 연동](lessons/06_ocr_ai_integration.md) | [실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/06_ocr_ai_integration.ipynb) |
| 7 | [추출 결과 검증 및 데이터 저장](lessons/07_validation_export.md) | [실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/07_validation_export.ipynb) |
| 8 | [실무 적용 시나리오 설계 및 최종 정리](lessons/08_business_application.md) | [실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/08_business_application.ipynb) |

Colab 링크는 `document_ai_lecture_2026` 브랜치를 가리킵니다.

## 안전하게 실습하기

- 식별정보를 가린 공개 샘플 또는 비식별 문서 한 장만 사용합니다.
- 개인 영수증을 사용한다면 카드번호·승인번호·현금영수증 번호·전화번호·회원번호를 먼저 가립니다.
- 실제 회사 문서와 개인 문서를 외부 API에 보내지 않습니다.
- 공개 Streamlit 터널이나 배포 주소를 만들지 않습니다.
- 준비 결과는 실제 모델 결과와 구분해 표시합니다.
- 원문에 없는 값은 추측하지 않고 `null`로 둡니다.
- 오류가 있거나 사람이 승인하지 않았다면 Excel을 만들지 않습니다.
- 수업이 끝나면 Colab 출력과 런타임 파일을 삭제하고 런타임을 종료합니다.

## 실행이 막혔을 때

```text
3분이 지나면 실행 중지
→ 준비 결과 선택
→ 필요한 셀을 위에서 아래로 다시 실행
→ 계속 실패하면 오류 화면을 닫지 말고 강사에게 알리기
```

자세한 내용은 [수강생 실습 환경](docs/environment.md)과 [수강생 문제 해결](docs/troubleshooting.md)에서 확인할 수 있습니다.

## 시작하기

1. [1교시 교재](lessons/01_document_ai_overview.md)를 엽니다.
2. 교재 상단의 Colab 버튼을 눌러 노트북을 엽니다.
3. 셀을 위에서 아래로 실행하고, 빈칸이 막히면 전체 정답으로 복구합니다.
4. 교시가 끝날 때 산출물 한 개를 내려받습니다.

각 노트북은 독립적으로 실행할 수 있으며 이전 교시 산출물의 완성 복구본을 포함합니다.

## 참고자료

교재는 2026-07-27 기준 공식 문서·표준·규제기관 자료를 중심으로 최소 10개 이상의 자료를 교차 검토했습니다. 어떤 자료가 어느 교시에 쓰였는지는 [과정 참고자료와 적용 범위](docs/course_references.md)에서 확인할 수 있습니다.

## 강사·교재 유지보수자 전용

아래 문서는 수강생 필수 학습 범위가 아닙니다.

- [강사용 과정 운영안](instructor/course_operation.md)
- [강사용 환경·모델 운영](instructor/environment_and_models.md)
- [강사용 복구·안전 대응](instructor/recovery_and_safety.md)
- [전체 커리큘럼 원전](instructor/curriculum.md)
- [교재 검증 보고서](docs/verification_report.md)
