# 4교시. 멀티모달·생성형 AI 기반 핵심 정보 추출

> 비식별 문서의 VLM 처리 예를 확인한 뒤, 준비된 구조 초안에서 원문 근거가 있는 값만 업무용 JSON으로 옮깁니다.
>
> **핵심 메시지:** VLM은 복잡한 문서 구조의 초안을 빠르게 만들 수 있지만, 원문에 없는 값을 만들 수 있으므로 검증 전에는 확정 데이터가 아닙니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/04_genai_extraction.ipynb)

## 1. 학습 목표

- 영수증 필드의 이름·자료형·필수 여부를 JSON 스키마로 표현할 수 있다.
- 준비된 VLM 구조 초안에서 각 값의 원문 근거를 찾을 수 있다.
- 찾을 수 없는 값은 추측하지 않고 `null`로 처리할 수 있다.

## 2. 이번 교시의 결과물

- `receipt.json`: 상호명·날짜·품목·합계와 원문 근거를 담은 검토용 JSON

## 3. 시작하기 전에

### 선수 지식

- Python 딕셔너리와 Markdown 표의 모양을 알면 충분하다.

### 준비 파일

- 3교시의 `clean_receipt.json` 또는 교재에 포함된 완성 복구본
- [비식별 한국 실물 영수증](../sample_docs/public_receipts/korea/taebaek_restaurant_2025_redacted.png)
- [PaddleOCR-VL 준비 Markdown](../sample_outputs/paddleocr_vl_result.md)
- [PaddleOCR-VL 준비 블록](../sample_outputs/paddleocr_vl_result.json)
- [준비 추출 결과](../sample_outputs/extracted_result.json)
- [4교시 Colab 노트북](../colab/04_genai_extraction.ipynb)

이번 시간에는 API 키나 과금 없이 준비된 VLM 구조 초안을 사용합니다. 초안의 각 값을 원본과 비교하고, 찾을 수 없는 값은 `null`로 바꿉니다.

## 4. 핵심 개념

### 4.1 VLM은 이미지와 문서 구조를 함께 본다

OCR 결과에서는 주로 인식 텍스트와 위치를 확인합니다. VLM(Vision-Language Model)은 이미지와 지시문을 함께 받아 제목·문단·표를 포함한 구조 초안을 만들 수 있습니다. 일부 VLM은 글자 인식도 함께 시도하지만 결과는 반드시 원본과 비교합니다.

> **쉬운 비유**
> OCR은 종이에 적힌 낱말을 받아 적는 사람이고, 문서 VLM은 제목과 표의 칸까지 살펴 초안을 정리하는 사람이다.

비유의 한계: VLM도 내용을 확정하는 담당자가 아니다. 보이지 않는 값을 만들거나 표의 숫자를 틀릴 수 있다.

![영수증이 PaddleOCR-VL을 거쳐 Markdown 구조로 바뀌는 흐름](assets/04/01_receipt_to_json.svg)

### 4.2 모델 결과와 업무 JSON은 다르다

PaddleOCR-VL은 Markdown과 레이아웃 블록을 반환할 수 있다. 회사 시스템이 원하는 `store_name`, `date`, `items`, `total_amount`와는 모양이 다르므로 변환 규칙이 필요하다.

```text
문서 이미지 → PaddleOCR-VL → Markdown·블록 → 업무 JSON
```

### 4.3 스키마와 원문 검증은 여전히 필요하다

![VLM 중간 결과, JSON 스키마, 원문 근거를 차례로 확인하는 그림](assets/04/02_three_checks.svg)

1. 구조: 필요한 네 필드가 있는가?
2. 자료형: 총액은 정수이고 품목은 배열인가?
3. 근거: 값이 실제 문서와 같은가?

원문에서 찾지 못한 값은 추측하지 않고 `null`로 둔다.

## 5. 전체 실습 흐름

```text
비식별 문서의 VLM 처리 전후 비교
  → 준비된 VLM 구조 초안 읽기
  → 제목·표·합계 찾기
  → 네 필드의 JSON으로 변환
  → 자료형과 원문 근거 확인
  → receipt.json 저장
```

## 6. 단계별 실습

### 실습 1. Markdown 중간 결과를 JSON으로 바꾸기

노트북이 제공하는 `SAMPLE_VLM_MARKDOWN`에는 제목, 날짜, 품목 표, 합계가 있다.

```python
receipt = extract_prepared_result(SAMPLE_VLM_MARKDOWN)

assert receipt["store_name"] == "샘플문구점"
assert receipt["total_amount"] == 5000
assert len(receipt["items"]) == 2
```

**기대 결과**

```json
{
  "store_name": "샘플문구점",
  "date": "2026-07-27",
  "total_amount": 5000,
  "items": [
    {"name": "연필", "quantity": 2, "line_total": 2000},
    {"name": "노트", "quantity": 1, "line_total": 3000}
  ]
}
```

**준비 결과 경로**

준비된 Markdown을 같은 변환 함수에 넣습니다. 결과의 `source_mode`가 `prepared_vlm`인지 확인해 실제 실행 결과와 구분합니다. 실행이 3분을 넘으면 중지하고 필요한 셀을 위에서 아래로 다시 실행합니다. 계속 실패하면 오류 화면을 닫지 말고 강사에게 알립니다.

## 7. 실습 결과 확인

- VLM의 Markdown과 최종 업무 JSON을 구분했는가?
- 네 필드의 값마다 원문에서 확인한 근거가 있는가?
- 원문에서 찾을 수 없는 값은 `null`인가?
- `receipt.json`을 Colab에서 내려받았는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 품목이 비어 있음 | Markdown 표 구분자 처리 오류 | `|`로 나눈 셀 네 개 확인 |
| 변환 실행이 멈춤 | 셀 실행 순서 또는 입력 누락 | 준비 결과 셀부터 위에서 아래 재실행 |
| JSON은 맞지만 값이 다름 | 원문 근거 검사 누락 | 합성 영수증과 날짜·총액 비교 |

## 9. 형성평가

1. PaddleOCR-VL의 Markdown이 바로 업무 JSON인가?
2. 원문에서 날짜를 찾지 못했다면 무엇을 넣는가?

<details>
<summary>정답 보기</summary>

1. 아니다. 업무 스키마에 맞게 변환하고 검증해야 한다.
2. 추측하지 않고 `null`을 넣는다.

</details>

## 10. 핵심 요약

- 문서 VLM은 이미지와 배치를 함께 처리한다.
- Markdown·레이아웃 결과는 업무 JSON 전의 중간 결과다.
- 스키마·자료형·원문 근거를 따로 확인한다.

## 11. 완료 체크리스트

- [ ] OCR 결과와 VLM 구조 초안의 차이를 설명했다.
- [ ] Markdown 표를 네 필드의 JSON으로 바꿨다.
- [ ] `receipt.json`을 만들었다.

## 12. 다음 교시 예고

5교시에서는 파일 입력과 판독 원문·JSON 결과를 보여 주는 작은 Streamlit 앱 코드를 Colab에서 만든다.

## 참고 자료

- [PaddleOCR-VL 1.6 모델 설명](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)
- [PaddleOCR-VL 파이프라인 사용법](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [JSON Schema 소개](https://json-schema.org/understanding-json-schema/about)
- [과정 참고자료와 적용 범위](../docs/course_references.md)
