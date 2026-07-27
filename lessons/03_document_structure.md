# 3교시. OCR 초안을 정돈된 데이터로 바꾸기

> OCR 원문을 보존하면서 헤더·날짜·품목·합계 줄로 정리합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/03_document_structure.ipynb)

## 1. 학습 목표

- 키-값과 반복 항목의 차이를 설명할 수 있다.
- OCR 텍스트의 공백과 빈 줄을 안전하게 정리할 수 있다.
- 원문·정제 결과·변경 기록을 함께 보존할 수 있다.

## 2. 이번 교시의 결과물

- `clean_receipt.json`: `raw_text`, `cleaned_lines`, `groups`, `change_log`가 포함된 파일

## 3. 시작하기 전에

### 선수 지식

- 문자열의 `strip()`, `splitlines()`를 본 적이 있으면 충분하다.

### 준비 파일

- [OCR 결과 텍스트](../sample_outputs/ocr_result.txt)
- [3교시 Colab 노트북](../colab/03_document_structure.ipynb)

## 4. 핵심 개념

### 4.1 키-값과 반복 항목은 구조가 다르다

- `거래일자: 2026-07-27`은 하나의 키와 값이다.
- 품목은 같은 구조가 여러 줄 반복되는 목록이다.
- 합계는 문서에서 한 번 나타나는 중요 필드다.

![영수증을 헤더, 날짜, 반복 품목, 합계로 나눈 구조 지도](assets/03/02_structure_map.svg)

### 4.2 OCR 줄 순서가 읽기 순서와 다를 수 있다

복잡한 문서에서는 위아래나 좌우 순서가 섞일 수 있다. 이번 합성 영수증은 단순한 예제만 사용하고 복잡한 표 복원은 다루지 않는다.

### 4.3 정제는 없는 값을 만들지 않는다

> **쉬운 비유**
> OCR 텍스트는 장바구니에 섞인 물건이고 정제는 물건을 종류별 바구니에 나누는 작업이다.

비유의 한계: 실제 복잡한 표는 단순 줄 나누기만으로 복원되지 않을 수 있다.

![불규칙한 공백이 있는 OCR 원문과 정제 결과 비교](assets/03/01_clean_before_after.svg)

원문을 덮어쓰지 않고 다음 세 가지를 함께 보존한다.

- `raw_text`: PaddleOCR가 반환한 원문
- `cleaned_lines`: 공백과 빈 줄을 정리한 결과
- `change_log`: 무엇을 바꿨는지 기록

## 5. 전체 실습 흐름

```text
SAMPLE_OCR_TEXT
  → 줄 단위 분리
  → 공백 정리
  → 헤더·날짜·품목·합계 분류
  → 원문과 변경 기록 함께 저장
```

## 6. 단계별 실습

### 실습 1. 원문을 보존하며 줄 정리하기

노트북에 제공된 `normalize_line()`에서 공백을 정리하는 핵심 한 줄을 읽고 합성 OCR 텍스트에 적용한다.

```python
import re

def normalize_line(line):
    original = line
    cleaned = re.sub(r"\s+", " ", line.strip())
    changes = []
    if original != cleaned:
        changes.append(f"공백 정리: {original!r} → {cleaned!r}")
    return cleaned, changes
```

완성 코드의 `group_receipt_lines()`가 줄을 네 영역으로 분류한다.

**기대 결과**

- 품목 그룹에 연필과 노트 두 줄이 들어간다.
- `raw_text`가 그대로 남는다.
- 불규칙한 공백을 바꿨다면 `change_log`에 기록된다.

**mock 대체 경로**

파일을 찾지 못해도 노트북의 `SAMPLE_OCR_TEXT`를 같은 함수에 입력한다. 정제 단계를 건너뛰지 않는다.

## 7. Codex 활용

### 요청 목표

정제 함수가 원문에 없는 값을 추가하지 않는지 검토한다.

### 실습 프롬프트

```text
목표: OCR 텍스트 정제 함수의 범위를 검토해줘.
맥락: 공백과 빈 줄만 정리하고 raw_text를 보존해야 해.
제약조건: 오타를 추측해서 고치거나 없는 값을 추가하지 마.
완료 기준: 위험한 변환이 있으면 해당 줄과 이유만 알려줘.
```

### 생성 결과 확인

- 원문이 결과에서 사라지지 않았는가?
- 오탈자를 마음대로 보정하지 않았는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 품목 그룹이 비어 있음 | `×`와 `개` 조건 누락 | 샘플 줄에 두 문자가 있는지 확인 |
| 원문과 정제 결과가 모두 바뀜 | 같은 변수를 덮어씀 | `raw_text`와 `cleaned_lines`를 분리 |
| 외부 파일이 없음 | Colab에 업로드하지 않음 | 내장 `SAMPLE_OCR_TEXT` 사용 |

## 9. 형성평가

1. 원문에서 찾지 못한 상호명을 정제 단계에서 추측해도 되는가?
2. 품목이 단순 키-값과 다른 이유는 무엇인가?

<details>
<summary>정답 보기</summary>

1. 안 된다. 정제는 원문 표현을 정돈하는 단계다.
2. 같은 구조의 품목이 여러 줄 반복되기 때문이다.

</details>

## 10. 핵심 요약

- 문서는 키-값과 반복 항목으로 나누어 볼 수 있다.
- 정제는 공백과 구조를 정리하는 일이다.
- 원문과 변경 기록을 함께 남긴다.

## 11. 완료 체크리스트

- [ ] 네 문서 영역을 구분했다.
- [ ] 원문과 정제 결과를 함께 보존했다.
- [ ] `clean_receipt.json`을 만들었다.

## 12. 다음 교시 예고

4교시에서는 PaddleOCR-VL이 표와 제목을 표현하는 방법을 보고, 그 결과를 업무용 JSON으로 바꾼다.

## 참고 자료

- [Amazon Textract 문서 분석 구조](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-analyzing.html)
- [Amazon Textract 표 구조](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html)
- [Google Document AI 응답 처리](https://docs.cloud.google.com/document-ai/docs/handle-response)
