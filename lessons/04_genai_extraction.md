# 4교시. 필요한 값만 JSON으로 담기

> 스키마에 맞춰 영수증 값을 JSON으로 만들고 원문에 없는 값은 `null`로 처리합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/04_genai_extraction.ipynb)

## 1. 학습 목표

- 영수증 JSON 스키마의 역할을 설명할 수 있다.
- 원문에 없는 값을 `null`로 표시할 수 있다.
- JSON 문법·자료형·원문 근거 검사를 구분할 수 있다.

## 2. 이번 교시의 결과물

- `receipt.json`: 네 필드와 품목 목록이 포함된 구조화 결과

## 3. 시작하기 전에

### 선수 지식

- Python 딕셔너리와 JSON의 모양을 구분할 수 있으면 충분하다.

### 준비 파일

- [OCR 결과 텍스트](../sample_outputs/ocr_result.txt)
- [mock 추출 결과](../sample_outputs/extracted_result.json)
- [4교시 Colab 노트북](../colab/04_genai_extraction.ipynb)

필수 실습은 API 키가 필요 없다. 실제 생성형 AI 호출은 선택 실습이다.

## 4. 핵심 개념

### 4.1 스키마는 데이터 설계도다

스키마는 어떤 필드가 있고, 값의 자료형은 무엇이며, 어떤 필드가 필수인지 정한다.

> **쉬운 비유**
> JSON 스키마는 빈 신청서 양식이고 추출은 영수증에서 찾은 값을 각 칸에 옮겨 적는 일이다.

비유의 한계: 양식의 모든 칸이 채워졌다고 내용까지 정확하다는 뜻은 아니다.

![영수증 네 영역에서 JSON 필드로 값이 이동하는 그림](assets/04/01_receipt_to_json.svg)

### 4.2 원문에 없으면 `null`이다

추출 시스템은 빈칸을 그럴듯한 값으로 채우면 안 된다.

```json
{
  "date": null
}
```

`null`은 오류를 숨기지 않고 “이 값을 찾지 못했다”라고 표시하는 방법이다.

### 4.3 형식과 사실성은 다른 검사다

![JSON 문법, 자료형, 원문 근거의 세 단계 검사](assets/04/02_three_checks.svg)

1. JSON 문법: 괄호와 쉼표가 올바른가?
2. 자료형: 총액이 정수이고 품목이 배열인가?
3. 원문 근거: 값이 실제 영수증과 같은가?

JSON Schema는 앞의 두 검사를 도울 수 있지만 세 번째 검사를 보장하지 않는다.

## 5. 전체 실습 흐름

```text
OCR 텍스트
  → 추출 프롬프트 확인
  → mock_extract 실행
  → JSON 스키마 대조
  → 원문과 총액 비교
  → receipt.json 저장
```

## 6. 단계별 실습

### 실습 1. JSON 구조와 `null` 규칙 확인하기

노트북에서 날짜와 합계 필드의 스키마 골격을 완성한다.

```python
RECEIPT_SCHEMA = {
    "date": {"type": ["string", "null"]},
    "total_amount": {"type": ["integer", "null"], "minimum": 0},
}
```

제공된 `mock_extract()`는 수업용 합성 영수증만 규칙으로 처리한다. 생성형 AI의 성능을 흉내 내는 함수가 아니다.

```python
result = mock_extract(SAMPLE_OCR_TEXT)
print(result["date"], result["total_amount"])
```

**기대 결과**

```text
2026-07-27 5000
```

**mock 대체 경로**

정규식 또는 스키마 패키지 오류가 나면 제공된 `extracted_result.json`을 읽고 필드와 원문을 눈으로 대조한다.

### 선택 실습. 실제 생성형 AI API

- Colab Secrets에 키가 있는 학습자만 실행한다.
- 실제 개인정보 문서를 보내지 않는다.
- 조직 적용 전 데이터 보존·학습 이용·리전·계약 조건을 확인한다.

선택 셀의 기본값은 `RUN_OPTIONAL_API = False`다.

## 7. Codex 활용

### 요청 목표

추출 프롬프트가 없는 값을 추측하지 않도록 개선한다.

### 실습 프롬프트

```text
목표: 영수증 OCR 텍스트를 JSON으로 추출하는 프롬프트를 검토해줘.
맥락: store_name, date, items, total_amount 네 필드만 사용해.
제약조건: 원문에 없는 값은 반드시 null, JSON 외 설명은 금지.
완료 기준: 누락된 제약조건이 있으면 한 줄씩 제안해줘.
```

### 생성 결과 확인

- 요청하지 않은 필드가 추가되지 않았는가?
- 원문에 없는 값을 `null`로 처리하는가?
- JSON 모양과 값의 정확성을 혼동하지 않는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| JSON 오류 | 따옴표·쉼표 누락 | 제공된 mock JSON과 모양 비교 |
| 날짜가 없음 | 원문에서 찾지 못함 | 추측하지 말고 `null` 유지 |
| API 키 오류 | Secrets 미설정 또는 권한 문제 | 선택 셀을 건너뛰고 mock 결과 사용 |

## 9. 형성평가

1. 원문에서 날짜를 찾지 못했다면 어떤 값을 넣는가?
2. JSON Schema 검사를 통과하면 내용도 정확한가?

<details>
<summary>정답 보기</summary>

1. `null`
2. 아니다. 문법과 자료형이 맞아도 원문 근거는 별도로 확인해야 한다.

</details>

## 10. 핵심 요약

- 스키마는 추출 결과의 모양을 먼저 정한다.
- 원문에 없는 값은 `null`로 둔다.
- JSON 문법·자료형·원문 근거는 서로 다른 검사다.

## 11. 완료 체크리스트

- [ ] 영수증 JSON 스키마를 읽었다.
- [ ] mock 결과와 원문을 비교했다.
- [ ] `receipt.json`을 만들었다.

## 12. 다음 교시 예고

5교시에서는 JSON 결과를 코드 출력이 아닌 Gradio 화면에 표시한다.

## 참고 자료

- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [OpenAI API 인증](https://platform.openai.com/docs/api-reference/authentication)
- [JSON Schema 2020-12](https://json-schema.org/specification)
