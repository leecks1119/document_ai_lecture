# 7교시. 추출 결과 검증 및 데이터 저장

> 날짜·금액·품목 합계를 검사하고 사람이 원본을 확인한 값만 업무용 Excel로 저장합니다.
>
> **핵심 메시지:** AI가 만든 JSON은 검증과 사람 확인을 통과해야 비로소 사용할 수 있는 업무 데이터가 됩니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/07_validation_export.ipynb)

## 1. 학습 목표

- `valid`, `warnings`, `errors`를 구분할 수 있다.
- 필수값·날짜 형식·금액·품목 합계를 검사할 수 있다.
- 오류가 없는 결과만 안전한 `receipt_result.xlsx`로 저장할 수 있다.

## 2. 이번 교시의 결과물

- `receipt_result.xlsx`: 원문·정제값·최종값·검토 상태와 품목 행을 담은 Excel 파일

## 3. 시작하기 전에

### 선수 지식

- `if`, 리스트, 딕셔너리를 읽을 수 있으면 충분하다.

### 준비 파일

- 6교시의 통합 프로토타입 또는 교재에 포함된 완성 복구본
- [정상 JSON](../sample_outputs/extracted_result.json)
- [7교시 Colab 노트북](../colab/07_validation_export.ipynb)
- 저장소의 `src/validate.py`, `src/export.py`

테스트 데이터와 Excel 보조 함수는 제공된다. 학습자는 필수값과 품목 합계 규칙을 확인하고, 오류가 있을 때 다운로드가 차단되는지 직접 시험한다. `=`, `+`, `-`, `@`로 시작하는 추출 문자열은 Excel 수식으로 실행되지 않도록 보호한다.

## 4. 핵심 개념

### 4.1 정상·경고·오류는 다음 행동이 다르다

![정상은 저장, 경고는 확인, 오류는 수정으로 이어지는 검증 신호등](assets/07/01_validation_signal.svg)

- `valid`: 현재 규칙에서 저장 가능
- `warnings`: 사람 확인 후 진행 가능
- `errors`: 수정 전에는 저장하면 안 됨

### 4.2 자료형과 업무 규칙은 다르다

총액 `6000`은 정수라서 자료형은 맞지만, 품목 합계 `5000`과 다르므로 업무 규칙 오류다.

> **쉬운 비유**
> 검증 함수는 정해진 규칙에 걸리는 항목을 찾는 공항 보안 검색대와 같다.

비유의 한계: 검색대를 통과했다고 내용의 의미와 사용 목적까지 올바른 것은 아니다. 중요한 값은 원문과 비교한다.

### 4.3 품목 하나가 Excel의 한 행이 된다

![JSON의 품목 배열이 Excel의 두 행으로 바뀌는 그림](assets/07/02_json_to_excel.svg)

상호명·날짜·총액은 각 품목 행에 반복하고, 품목명·수량·단가·소계는 행마다 달라진다.

## 5. 전체 실습 흐름

```text
제공된 검증용 JSON 다섯 개
  → 필수값 검사
  → 품목 합계 검사
  → 원본 대조와 사람 확인
  → 오류가 없는 결과 선택
  → receipt_result.xlsx 생성
  → Excel 다운로드
```

## 6. 단계별 실습

### 실습 1. 검증과 사람 승인 뒤 Excel 만들기

노트북이 제공하는 정상·상호명 누락·날짜 형식·금액 형식·합계 불일치 데이터를 완성된 규칙에 차례로 넣어 결과를 비교한다.

```python
def validate_receipt(data):
    errors = []

    for field in ("store_name", "date", "total_amount", "items"):
        if data.get(field) in (None, "", []):
            errors.append(f"필수값 누락: {field}")

    if data.get("date") and not is_iso_date(data["date"]):
        errors.append("date는 YYYY-MM-DD 형식이어야 합니다.")

    total_amount = data.get("total_amount")
    if total_amount is not None and (
        not isinstance(total_amount, int) or total_amount < 0
    ):
        errors.append("total_amount는 0 이상의 정수여야 합니다.")

    item_sum = sum(item["line_total"] for item in data.get("items", []))
    if isinstance(total_amount, int) and total_amount != item_sum:
        errors.append("품목 합계와 총액이 다릅니다.")

    return {"valid": not errors, "warnings": [], "errors": errors}
```

**기대 결과**

- 정상 데이터: `valid=True`
- 상호명 누락: `필수값 누락: store_name`
- 날짜·금액 형식 오류: 각각 형식 오류가 표시됨
- 합계 불일치: `품목 합계와 총액이 다릅니다.`

Excel 생성과 수식 문자 보호 함수는 완성 코드로 제공된다. 이 함수는 `validation["valid"]`와 `human_approved`를 모두 확인한다. 하나라도 `False`면 파일을 만들지 않는다.

원본과 세 값을 대조한 뒤에만 사람 승인 표시를 바꾼다.

```python
HUMAN_APPROVED = True  # 원본과 추출값을 직접 확인한 뒤에만 True
assert HUMAN_APPROVED, "원본을 확인하고 승인해야 Excel을 생성합니다."
```

생성된 파일은 다음 세 시트를 가진다.

| 시트 | 남기는 내용 |
| --- | --- |
| `검토_요약` | 필드별 원문값·정제값·최종값·검토 상태 |
| `품목` | 반복 품목 한 개당 한 행 |
| `원문` | 사용 모드와 OCR·판독 원문 |

노트북은 합계 오류 데이터와 사람 미승인 데이터에서 Excel 파일이 생기지 않는지도 확인한다.

**mock 대체 경로**

OCR·VLM 모델이 없어도 노트북의 `SAMPLE_RECEIPT`를 같은 검증 함수에 전달한다. Streamlit 화면을 실행하지 못해도 Colab의 `files.download()`로 결과를 받는다.

## 7. 실습 결과 확인

- 정상·필수값 누락·날짜·금액·합계 오류 데이터의 결과가 서로 다른가?
- 오류가 있으면 `valid=False`이고 Excel 생성이 차단되는가?
- 원본과 추출값을 대조한 뒤에만 사람 승인 표시를 바꿨는가?
- 원문·최종값·검토 상태가 Excel에 남는가?
- `검토_요약`·`품목`·`원문` 세 시트가 있는가?
- `receipt_result.xlsx`를 Colab에서 내려받아 직접 열어 봤는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 정상 데이터도 합계 오류 | `line_total` 대신 단가를 더함 | 품목의 `line_total` 합 확인 |
| Excel 파일이 만들어지지 않음 | 검증 오류가 남아 있음 | 오류 필드 수정 후 원문과 다시 비교 |
| 셀이 수식처럼 보임 | 추출 문자열이 `=`, `+`, `-`, `@`로 시작 | 제공 수식 실행 방지 함수 적용 |
| 다운로드 버튼 확인이 어려움 | Streamlit 화면 미실행 | Colab `files.download()` 사용 |

## 9. 형성평가

1. 총액 자료형이 정수여도 품목 합계와 다르면 어떻게 처리하는가?
2. 날짜 형식이 맞으면 원문과도 같다고 확정할 수 있는가?

<details>
<summary>정답 보기</summary>

1. 이번 규칙에서는 `error`로 표시하고 저장 전에 수정한다.
2. 아니다. 원문과 다시 비교한다.

</details>

## 10. 핵심 요약

- 검증 결과는 정상·경고·오류로 나눈다.
- 자료형 검사와 업무 규칙 검사는 다르다.
- 반복 품목은 Excel 품목 시트의 여러 행으로 펼친다.
- 원문·정제값·최종값·검토 상태를 함께 남긴다.

## 11. 완료 체크리스트

- [ ] 필수값과 품목 합계 규칙을 실행했다.
- [ ] 다섯 테스트 데이터의 결과를 비교했다.
- [ ] 오류 결과의 다운로드가 차단되는지 확인했다.
- [ ] `receipt_result.xlsx`를 만들었다.

## 12. 다음 교시 예고

8교시에서는 새 기능을 추가하지 않고 견적서·신청서·거래명세서 중 하나의 PoC 후보를 검토한다.

## 참고 자료

- [Google Document AI 평가](https://docs.cloud.google.com/document-ai/docs/evaluate)
- [OWASP CSV Injection](https://owasp.org/www-community/attacks/CSV_Injection)
