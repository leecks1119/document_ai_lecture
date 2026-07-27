# 3교시. 문서 구조 이해 및 추출 결과 정제

> OCR 원문을 지우지 않고 상호명·날짜 같은 키-값과 품목처럼 반복되는 행을 다시 묶습니다.
>
> **핵심 메시지:** 문서 자동화의 핵심은 글자를 읽은 뒤 업무에 필요한 관계와 반복 행을 다시 구성하는 것입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/03_document_structure.ipynb)

## 1. 학습 목표

- 키-값과 반복 항목의 차이를 설명할 수 있다.
- OCR 원문을 보존하면서 공백·빈 줄·읽기 순서를 정리할 수 있다.
- 품목을 한 줄에 한 항목씩 반복 행으로 재구성할 수 있다.

## 2. 이번 교시의 결과물

- `clean_receipt.json`: `raw_text`, `cleaned_lines`, `groups`, `change_log`가 포함된 파일

## 3. 시작하기 전에

### 선수 지식

- 문자열의 `strip()`, `splitlines()`를 본 적이 있으면 충분하다.

### 준비 파일

- 2교시의 `ocr_result.json` 또는 교재에 포함된 완성 복구본
- [OCR 결과 텍스트](../sample_outputs/ocr_result.txt)
- [3교시 Colab 노트북](../colab/03_document_structure.ipynb)

Excel·Word·PPT 원본에는 셀·문단·도형 구조가 남아 있을 수 있고, 텍스트 PDF는 글자를 직접 선택할 수 있습니다. 표 캡처·사진·스캔은 픽셀만 남으므로 OCR이나 VLM을 검토합니다. 이번 시간에는 이 차이를 비교한 뒤 영수증의 키-값과 반복 품목 행을 직접 정리합니다.

## 4. 핵심 개념

### 4.1 키-값과 반복 항목은 구조가 다릅니다

- `거래일자: 2026-07-27`은 하나의 키와 값입니다.
- 품목은 같은 구조가 여러 줄 반복되는 목록입니다.
- 합계는 문서에서 한 번 나타나는 중요 필드입니다.

![영수증을 헤더, 날짜, 반복 품목, 합계로 나눈 구조 지도](assets/03/02_structure_map.svg)

### 4.2 OCR 줄 순서가 읽기 순서와 다를 수 있다

복잡한 문서에서는 위아래나 좌우 순서가 섞일 수 있습니다. 특히 표 캡처는 행과 열 정보가 사라져 글자를 읽은 뒤 관계를 다시 만들어야 합니다. 이번 시간에는 영수증의 반복 품목 행까지만 직접 정리하고, 복잡한 표는 준비된 결과에서 구조만 확인합니다.

### 4.3 정제는 없는 값을 만들지 않습니다

> **쉬운 비유**
> OCR 텍스트는 장바구니에 섞인 물건이고 정제는 물건을 종류별 바구니에 나누는 작업과 같습니다.

비유의 한계: 실제 복잡한 표는 단순 줄 나누기만으로 복원되지 않을 수 있습니다.

![불규칙한 공백이 있는 OCR 원문과 정제 결과 비교](assets/03/01_clean_before_after.svg)

원문을 덮어쓰지 않고 다음 세 가지를 함께 보존합니다.

- `raw_text`: PaddleOCR이 읽은 원문
- `cleaned_lines`: 공백과 빈 줄을 정리한 결과
- `change_log`: 무엇을 바꿨는지 기록

## 5. 전체 실습 흐름

```text
2교시 OCR 결과 또는 완성 복구본
  → 줄 단위 분리
  → 공백 정리
  → 헤더·날짜·품목·합계 분류
  → 원문과 변경 기록 함께 저장
```

## 6. 단계별 실습

### 실습 1. 원문을 보존하며 줄 정리하기

노트북에 제공된 `normalize_line()`에서 공백을 정리하는 핵심 한 줄을 읽고 합성 OCR 텍스트에 적용합니다.

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

준비된 `group_receipt_lines()`를 실행해 줄을 헤더·날짜·품목·합계 영역으로 나눕니다.

**기대 결과**

- 품목 그룹에 연필과 노트 두 줄이 들어갑니다.
- `raw_text`가 그대로 남습니다.
- 불규칙한 공백을 바꿨다면 `change_log`에 기록됩니다.

**준비 결과 경로**

파일을 찾지 못하면 `준비 텍스트 사용`을 선택하고 `SAMPLE_OCR_TEXT`를 같은 함수에 입력합니다. 정제 단계는 건너뛰지 않습니다.

## 7. 실습 결과 확인

- `raw_text`가 정제 전 모습 그대로 남아 있는가?
- 공백과 빈 줄 외의 값을 추측해서 바꾸지 않았는가?
- 상호명·날짜·합계와 반복 품목이 서로 다른 그룹에 들어갔는가?
- `clean_receipt.json`을 Colab에서 내려받았는가?

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

1. 안 됩니다. 정제는 원문 표현을 정돈하는 단계입니다.
2. 같은 구조의 품목이 여러 줄 반복되기 때문입니다.

</details>

## 10. 핵심 요약

- 문서는 키-값과 반복 항목으로 나누어 볼 수 있습니다.
- 정제는 공백과 구조를 정리하는 일입니다.
- 원문과 변경 기록을 함께 남깁니다.

## 11. 완료 체크리스트

- [ ] 네 문서 영역을 구분했다.
- [ ] 원문과 정제 결과를 함께 보존했다.
- [ ] `clean_receipt.json`을 만들었다.

## 12. 다음 교시 예고

4교시에서는 PaddleOCR-VL이 표와 제목을 표현하는 방법을 보고, 그 결과를 업무용 JSON으로 바꿉니다.

## 참고 자료

- [Amazon Textract 문서 분석 구조](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-analyzing.html)
- [Amazon Textract 표 구조](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html)
- [Google Document AI 응답 처리](https://docs.cloud.google.com/document-ai/docs/handle-response)
- [과정 참고자료와 적용 범위](../docs/course_references.md)
