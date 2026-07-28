# 1교시. 한국 영수증으로 구분하는 OCR·VLM·Document AI

> **이번 교시의 한 문장:** 문서 자동화는 “글자를 읽는 모델” 하나가 아니라, 읽기·구조화·검증·사람 확인·업무 연결이 이어지는 과정입니다.

[Colab 실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/01_document_ai_overview.ipynb)

## 60분 뒤 남길 것

- OCR·Multimodal AI·VLM·Document AI·IDP를 자기 말로 구분합니다.
- 공개 한국 영수증의 값 하나가 원본 근거와 함께 Excel까지 가는 과정을 설명합니다.
- `course_outputs/receipt_pipeline_trace.json`을 만듭니다.

수업의 약 10%는 개념, 나머지는 실제 영수증을 보고 연결하는 활동입니다.

## 사용할 실제 문서

![개인정보와 거래 식별 영역을 가린 2025년 한국 음식점 영수증](../sample_docs/public_receipts/korea/taebaek_restaurant_2025_redacted.png)

이 문서는 Wikimedia Commons의 공개 영수증을 비식별 처리한 파생본입니다. 출처·라이선스·해시는 [`metadata.json`](../sample_docs/public_receipts/metadata.json)에 기록되어 있습니다.

## 개념 10%: 네 역할만 기억하기

### OCR: 글자와 위치를 읽는다

OCR은 이미지의 픽셀을 보고 텍스트를 인식합니다. 제품에 따라 텍스트와 함께 바운딩 박스, 페이지, 신뢰도를 반환합니다. OCR은 “이 글자가 무엇처럼 보이는가”를 다루며, 합계의 업무적 의미까지 보장하지 않습니다.

### Multimodal AI와 VLM: 이미지와 언어를 함께 본다

Multimodal AI는 이미지·텍스트 등 둘 이상의 형식을 다루는 상위 범주입니다. VLM은 그중 이미지와 언어를 함께 처리하는 모델입니다. 표·제목·문맥을 이용해 구조 초안을 만들 수 있지만 원문에 없는 값을 생성할 수도 있습니다.

### Document AI: 문서를 업무 데이터로 구조화한다

Document AI는 문서 분류, OCR, 레이아웃 분석, 필드·표 추출, 정규화, 검증 같은 문서 처리 능력을 묶어 부르는 말입니다. OCR이나 VLM을 선택하거나 조합할 수 있으며, 고정된 `OCR → VLM` 순서를 뜻하지 않습니다.

### IDP: 예외·사람 승인·업무 연결까지 운영한다

IDP(Intelligent Document Processing)는 Document AI 능력을 접수, 보안, 예외 처리, 사람 검토, ERP·Excel 연결, 관측과 개선에 결합한 운영 범위입니다.

![Multimodal AI, VLM, OCR, Document AI 역량과 IDP 운영 범위의 관계](assets/01/01_terms_relationship.svg)

> 쉬운 비유: OCR은 영수증을 받아 적는 사람, VLM은 표와 문맥을 보고 초안을 정리하는 사람, Document AI는 문서를 데이터로 바꾸는 도구 상자, IDP는 접수부터 결재·전달까지 정한 회사 업무 절차입니다.

## 전체 절차: 0~12 지도

실제 기업 구현에서는 단계를 합치거나 순서를 바꿀 수 있습니다. 여기서는 빠뜨리지 않기 위한 참조 지도로 사용합니다.

| 단계 | 질문 | 오늘의 영수증 예 |
| ---: | --- | --- |
| 0 | 어떤 업무 문제를 줄일 것인가? | 비용 입력 시간을 줄인다 |
| 1 | 성공을 무엇으로 판단할까? | 필드 정확도·수정률·처리시간 |
| 2 | 어떤 입력을 받을까? | 승인된 비식별 한 장 |
| 3 | 품질과 형식을 확인했나? | 흐림·기울기·PDF 페이지 수 |
| 4 | 문서 종류는 무엇인가? | 영수증 |
| 5 | 어떤 방식으로 읽을까? | 한국어 OCR 또는 문서 VLM |
| 6 | 어떤 구조를 찾을까? | 키-값·반복 품목·합계 |
| 7 | 어떤 스키마로 만들까? | 날짜 문자열·금액 정수·품목 배열 |
| 8 | 어떤 값이 맞는지 검사할까? | 수량×단가, 품목 합계, 날짜 |
| 9 | 예외는 어디로 보낼까? | 오류·경고·검토 대기 |
| 10 | 누가 최종 확인할까? | 비용 처리 담당자 |
| 11 | 어디에 저장할까? | 승인 후 Excel |
| 12 | 운영하면서 무엇을 개선할까? | 오류 원인·수정률·비용 |

보안과 개인정보는 한 단계가 아니라 전체 과정을 가로지릅니다.

![업무 목표부터 운영 평가까지 이어지는 Document AI 전체 참조 지도](assets/01/02_enterprise_pipeline.svg)

## 실습 90%: 값 하나의 여행 추적하기

### 1. Colab에서 공개 영수증을 확인합니다

노트북 첫 셀에서 `Python`, `공통 작업 폴더`, 문서 이미지를 확인합니다. 개인·회사 문서는 업로드하지 않습니다.

### 2. 합계 76,000원의 역할을 연결합니다

노트북의 `내 연결` 빈칸에 각 단계가 만드는 파일을 먼저 적습니다. 막히면 바로 아래 힌트를 보고, 마지막에는 공개된 전체 정답과 비교합니다.

```text
OCR: "합계 금액 76,000"을 읽음
VLM: 이 줄이 total_amount 후보라고 구조 초안을 만듦
Document AI: 76000 정수·품목 합계·원본 근거를 검사
IDP: 사람이 원본을 승인한 뒤 Excel로 전달
```

![영수증 원문에서 근거·정규화·검증·사람 확인을 거쳐 Excel로 가는 흐름](assets/01/03_receipt_evidence_flow.svg)

### 3. 처리 흔적 파일을 만듭니다

노트북을 위에서 아래로 실행합니다. 마지막에 다음 문구가 보여야 합니다.

```text
CHECKPOINT 1/1 PASS: course_outputs/receipt_pipeline_trace.json
```

파일 안의 핵심 구조는 다음과 같습니다.

```json
{
  "roles": [
    {"role": "OCR", "artifact": "ocr_result.json"},
    {"role": "VLM", "artifact": "receipt_draft.json"},
    {"role": "Document AI", "artifact": "validated_receipt.json"},
    {"role": "IDP", "artifact": "receipt_result.xlsx"}
  ],
  "evidence_example": {
    "field": "total_amount",
    "value": 76000,
    "source_text": "합계 금액 76,000",
    "decision": "REVIEW_BEFORE_EXPORT"
  }
}
```

빈칸을 틀려도 수업이 멈추지 않습니다. 자신의 답을 먼저 남긴 뒤 전체 정답 셀을 실행해 결과 파일을 복구합니다. 중요한 것은 문법 암기가 아니라 “어떤 기술이 어떤 결과를 다음 단계에 넘기는지” 설명하는 것입니다.

## 통과 기준

- `receipt_pipeline_trace.json`이 생성되었습니다.
- OCR·VLM·Document AI·IDP를 “읽기·초안·검증·업무 연결”로 설명할 수 있습니다.
- `76000`이 원본의 `"합계 금액 76,000"`에서 왔음을 찾을 수 있습니다.
- AI 결과가 바로 Excel로 저장되지 않고 사람 확인을 거친다고 설명할 수 있습니다.

## 막혔을 때

1. Colab 셀을 위에서 아래로 다시 실행합니다.
2. `course_outputs` 폴더가 만들어졌는지 확인합니다.
3. 3분 이상 멈추면 실행을 중지하고 오류 화면을 그대로 둡니다.
4. [`docs/troubleshooting.md`](../docs/troubleshooting.md)의 같은 오류 항목을 확인합니다.

## 1분 확인

1. OCR과 VLM의 가장 큰 차이는 무엇인가요?
2. Document AI와 IDP의 경계에 사람 승인과 업무 시스템 연결이 있는 이유는 무엇인가요?
3. 신뢰도가 높아도 합계 금액을 원본과 대조해야 하는 이유는 무엇인가요?

다음 교시에는 같은 공개 한국 영수증에 실제 PaddleOCR을 실행해 글자·위치·신뢰도를 확인합니다.

## 참고 자료

이 교시에 사용한 공식·원 출처는 [과정 참고자료와 적용 범위](../docs/course_references.md)의 1교시 표에서 확인할 수 있습니다.
