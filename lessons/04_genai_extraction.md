# 4교시. PaddleOCR-VL로 문서 구조 읽기

> 문서 전용 멀티모달 모델의 중간 결과를 업무용 JSON으로 바꿉니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/04_genai_extraction.ipynb)

## 1. 학습 목표

- OCR과 문서 전용 VLM의 차이를 설명할 수 있다.
- PaddleOCR-VL의 Markdown 결과에서 제목·표·합계를 찾을 수 있다.
- 중간 결과를 정해진 JSON으로 바꾸고 원문과 대조할 수 있다.

## 2. 이번 교시의 결과물

- `receipt.json`: VLM 중간 결과를 네 필드로 바꾼 업무용 JSON

## 3. 시작하기 전에

### 선수 지식

- Python 딕셔너리와 Markdown 표의 모양을 알면 충분하다.

### 준비 파일

- [합성 영수증](../sample_docs/receipt_sample.png)
- [PaddleOCR-VL mock Markdown](../sample_outputs/paddleocr_vl_result.md)
- [PaddleOCR-VL mock 블록](../sample_outputs/paddleocr_vl_result.json)
- [mock 추출 결과](../sample_outputs/extracted_result.json)
- [4교시 Colab 노트북](../colab/04_genai_extraction.ipynb)

기본 실습은 PaddleOCR-VL 형태의 준비된 Markdown을 사용한다. 실제 모델은 크고 다운로드 시간이 필요하므로 선택 실행한다.

## 4. 핵심 개념

### 4.1 VLM은 이미지와 문서 구조를 함께 본다

OCR은 주로 글자와 위치를 반환한다. VLM(Vision-Language Model)은 이미지와 언어를 함께 처리해 제목, 문단, 표 같은 관계를 표현한다.

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
PaddleOCR-VL 형태의 Markdown 읽기
  → 제목·표·합계 찾기
  → 네 필드의 JSON으로 변환
  → 자료형과 원문 근거 확인
  → receipt.json 저장
```

## 6. 단계별 실습

### 실습 1. Markdown 중간 결과를 JSON으로 바꾸기

노트북이 제공하는 `SAMPLE_VLM_MARKDOWN`에는 제목, 날짜, 품목 표, 합계가 있다.

```python
receipt = mock_extract(SAMPLE_VLM_MARKDOWN)

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

**mock 대체 경로**

모델을 실행하지 않아도 준비된 Markdown을 동일한 변환 함수에 넣는다. 결과의 `source_mode`가 `mock_vlm`인지 확인한다.

**선택: 실제 PaddleOCR-VL 1.6 실행**

```python
RUN_PADDLEOCR_VL = False
```

Colab에 충분한 메모리가 있을 때만 `True`로 바꾼다. 모델 결과가 곧 업무 정답이라는 뜻은 아니므로 반드시 합성 원본과 비교한다.

## 7. Codex 활용

### 요청 목표

VLM 중간 결과를 네 필드로만 변환하는 코드를 검토한다.

### 실습 프롬프트

```text
목표: PaddleOCR-VL Markdown을 영수증 JSON으로 바꾸는 함수를 검토해줘.
맥락: store_name, date, items, total_amount만 필요해.
제약조건: 원문에 없는 값은 null, 새 필드를 추가하지 마.
완료 기준: 표 파싱과 합계 변환에서 틀릴 수 있는 부분만 알려줘.
```

### 생성 결과 확인

- Markdown과 업무 JSON을 같은 것으로 취급하지 않았는가?
- 없는 값을 추측하지 않았는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 품목이 비어 있음 | Markdown 표 구분자 처리 오류 | `|`로 나눈 셀 네 개 확인 |
| 실제 모델이 느림 | 모델 다운로드·메모리 사용 | 선택 셀 중지 후 mock 경로 사용 |
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

- [ ] OCR과 문서 VLM의 차이를 설명했다.
- [ ] Markdown 표를 네 필드의 JSON으로 바꿨다.
- [ ] `receipt.json`을 만들었다.

## 12. 다음 교시 예고

5교시에서는 OCR과 VLM 처리기를 고를 수 있는 작은 Gradio 화면을 만든다.

## 참고 자료

- [PaddleOCR-VL 1.6 모델 설명](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)
- [PaddleOCR-VL 파이프라인 사용법](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
