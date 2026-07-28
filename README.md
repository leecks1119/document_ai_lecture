# Document AI와 생성형 AI를 활용한 문서 데이터 추출 실습

> Python을 한 번 사용해 본 일반 실무자가 공개 한국 영수증 한 장으로 작은 Document AI 프로토타입을 완성하는 하루 8교시 과정

![영수증 한 장이 판독·구조화·검증·사람 확인을 거쳐 업무 데이터가 되는 과정](lessons/assets/course_cover_v2.png)

## 이 강의가 남겨야 할 한 문장

> “영수증으로 해보니 원리를 알겠다. 우리 회사의 견적서나 신청서도 이런 식으로 자동화해볼 수 있겠는데?”

여러분은 식별정보를 가린 공개 한국 영수증 한 장을 판독하고, 필요한 값을 JSON으로 구조화하고, 원본과 대조해 검증한 뒤 `receipt_result.xlsx`로 내려받습니다. 마지막에는 같은 방식을 견적서·신청서·거래명세서에 적용할 수 있는지 실제 샘플로 검토합니다. 발주서는 이번 입문 과정에서 다루지 않습니다.

과정을 마치면 다음을 할 수 있습니다.

- OCR·Multimodal AI·VLM·Document AI·IDP의 역할과 한계를 설명합니다.
- 실제 문서 한 장으로 작동하는 작은 프로토타입을 직접 만듭니다.
- 자동화할 필드, 검증 규칙, 사람 확인, 오류 영향을 정합니다.
- 우리 업무에서 작게 시험할 PoC 후보를 검토합니다.

## 하루 동안 만들 결과

```text
식별정보를 가린 공개 한국 영수증 한 장
  → LIVE OCR 또는 명시적 PREPARED_FALLBACK
  → 키-값·반복 행 정리
  → 업무용 JSON
  → 규칙 검증
  → 원본 대조와 사람 승인
  → 사람이 값을 수정·재검증
  → receipt_result.xlsx
```

모든 필수 실습은 Google Colab에서 문서 한 장씩 진행합니다. 시작 코드·빈칸·힌트·전체 정답이 제공되며 API 키나 유료 API 결제가 필요하지 않습니다. 2교시는 실제 PaddleOCR 실행이 기본이며, 설치·모델 다운로드 장애 때만 준비 결과로 복구합니다.

## 8교시 커리큘럼

| 교시 | 공지된 주제 | 이번 교시의 한 가지 메시지 | 산출물 |
| --- | --- | --- | --- |
| 1 | 한국 영수증으로 구분하는 OCR·VLM·Document AI | 문서 자동화는 모델 하나가 아니라 목표부터 운영 개선까지 이어지는 과정입니다. | `receipt_pipeline_trace.json` |
| 2 | OCR 기반 텍스트 추출 실습 | OCR 결과는 정답이 아니라 원본과 대조할 판독 결과입니다. | `lesson02_ocr_outputs.zip` |
| 3 | 문서 구조 이해 및 추출 결과 정제 | 읽힌 글자를 키-값과 반복 행으로 재구성해야 업무 데이터가 됩니다. | `clean_receipt.json` |
| 4 | 멀티모달·생성형 AI 기반 핵심 정보 추출 | OCR+규칙과 VLM은 서로 다른 초안 경로이며 둘 다 근거 확인이 필요합니다. | `receipt.json`, `vlm_comparison.json` |
| 5 | 문서 자동화 웹 애플리케이션 기본 구현 | 처리 함수에 입력·실행·결과 화면을 붙이면 사용할 수 있는 도구가 됩니다. | `app_05.py` |
| 6 | OCR 및 정보 추출 기능 연동 | 파일에서 JSON까지 단계를 연결하면 실패 위치를 찾을 수 있습니다. | `app_06.py` |
| 7 | 추출 결과 검증 및 데이터 저장 | 사람이 추출값을 수정·재검증하고 승인한 값만 업무용 Excel이 됩니다. | `receipt_result.xlsx`, `final_document_ai_app.zip` |
| 8 | 실무 적용 시나리오 설계 및 최종 정리 | 같은 흐름을 재사용하되 PoC 적합성은 업무별로 판단합니다. | `poc_candidate_card.md`, `office_format_samples.zip` |

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

- Google Colab도 외부 클라우드이며, 파일 업로드는 외부 전송입니다.
- 식별정보를 가린 공개 샘플 또는 비식별 문서 한 장만 사용합니다.
- 필수 실습은 저장소의 공개·합성 샘플로 진행합니다.
- 개인 영수증은 조직 승인과 완전한 비식별화가 있는 선택 실습에서만 사용합니다.
- 조직 승인 없는 실제 회사 문서와 개인 문서를 Colab이나 외부 API에 보내지 않습니다.
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
4. 교시가 끝날 때 체크포인트 문구와 산출물 또는 ZIP 묶음을 확인합니다.

각 Colab 링크는 새 런타임과 새 파일시스템에서 열릴 수 있습니다. 2교시는 JSON과 판독 이미지를 ZIP 하나로 내려받고, 3·4교시는 다음 교시 입력 파일을 내려받습니다. 3·4·7교시 시작에는 이전 파일을 선택하는 업로드 창이 열립니다. 파일이 없거나 3분 안에 복구해야 할 때만 노트북의 `USE_PREPARED_INPUT=True`로 바꿉니다.

같은 런타임에 `course_outputs`가 남아 있으면 이전 산출물을 자동 사용합니다. 새 런타임에서도 다운로드→업로드로 자신의 결과를 이어 쓰며, 검수된 준비 입력은 명시적 복구 경로입니다.

5·6·7교시는 자동 검사뿐 아니라 Colab 안에서 Streamlit 화면을 직접 열어 조작할 수 있습니다. 7교시 최종 앱에서는 추출값과 품목 표를 수정하고 다시 검증한 뒤 승인해야 Excel 다운로드가 열립니다.

## 실제 형식 체험 파일

8교시에는 단순 설명이 아니라 다음 파일을 직접 엽니다.

- [견적서 Excel](sample_docs/formats/quotation.xlsx): 셀·수식·병합·숫자 서식
- [이미지 기반 신청서 Word](sample_docs/formats/application_form.docx): Word 안의 이미지 본문
- [거래명세서 PDF](sample_docs/formats/transaction_statement.pdf): 한 페이지 텍스트층
- [표 캡처 PowerPoint](sample_docs/formats/table_summary.pptx): 도형·읽기 순서·캡처 손실

교육용 합성 실물형 사진과 정답 JSON도 함께 제공합니다.

- [견적서 사진](sample_docs/extensions/quotation_photo.png)
- [신청서 사진](sample_docs/extensions/application_form_photo.png)
- [거래명세서 사진](sample_docs/extensions/transaction_statement_photo.png)

## 참고자료

교재의 기술 조사 기준일은 요청한 2026-07-27입니다. 공식 문서·표준·규제기관 자료를 중심으로 최소 10개 이상의 자료를 교차 검토했으며, 2026-07-28에는 링크와 구현 상태만 다시 확인했습니다. 어떤 자료가 어느 교시에 쓰였는지는 [과정 참고자료와 적용 범위](docs/course_references.md)에서 확인할 수 있습니다.

## 강사·교재 유지보수자 전용

아래 문서는 수강생 필수 학습 범위가 아닙니다.

- [강사용 과정 운영안](instructor/course_operation.md)
- [강사용 마스터 런북](instructor/master_runbook.md)
- [강사용 시연 플레이북](instructor/demo_playbook.md)
- [수강생 산출물 평가표](instructor/assessment_rubric.md)
- [20명 진행 현황판](instructor/cohort_progress_board.md)
- [성과 증명·업무 전이 운영안](instructor/outcome_evidence_plan.md)
- [강사용 환경·모델 운영](instructor/environment_and_models.md)
- [강사용 복구·안전 대응](instructor/recovery_and_safety.md)
- [전체 커리큘럼 원전](instructor/curriculum.md)
- [프리미엄 품질 감사](docs/premium_quality_audit.md)
- [교재 검증 보고서](docs/verification_report.md)
