# 7교시. 추출 결과 검증 및 데이터 저장

OCR과 VLM이 만든 값은 초안입니다. 총액이 틀리거나 날짜가 빠진 결과를 그대로
Excel에 저장하면 자동화가 새로운 업무 오류를 만들 수 있습니다.

> **이번 시간의 도착점:** 앞 교시 JSON을 검사하고, 원본 확인 뒤 세 시트가 있는
> Excel 파일을 실제로 내려받습니다.

[Colab 실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/master/colab/07_validation_export.ipynb)

![검증과 사람 확인 뒤 Excel로 저장하는 흐름](assets/07/01_validation_signal.svg)

## 자동으로 확인할 것

이번 실습에서는 다음 규칙만 사용합니다.

- 상호명·날짜·총액·품목이 비어 있지 않은가?
- 날짜가 `YYYY-MM-DD` 형식인가?
- 수량 × 단가가 품목 금액과 같은가?
- 품목 금액의 합과 영수증 총액이 같은가?
- 공급가액과 부가세의 합이 결제금액과 같은가?

규칙을 많이 만드는 것이 목적이 아닙니다. 틀린 값이 저장되기 전에 멈추는 경험이
핵심입니다.

## 직접 해보기

### 1. 앞 교시 결과를 검사합니다

앞 교시의 `receipt.json`이 있으면 그대로 사용하고, 없으면 공개 영수증 정답을
사용합니다.

```python
validation = validate_receipt(receipt)
```

화면에서 `검증 통과`, `오류`, `경고`를 확인합니다.

### 2. 승인 전 저장 차단을 확인합니다

다음 값이 `False`이면 Excel을 만들지 않습니다.

```python
원본_확인_완료 = False
```

화면에 `승인 전이므로 Excel 저장 차단`이 표시되는지 확인합니다.

### 3. 공개 정답으로 Excel을 만듭니다

수업에서는 공개 원본과 대조한 정답 경로를 바로 제공합니다. 결과 파일에는 다음
세 시트가 생깁니다.

- `검토_요약`: 원본값·최종값·검토자
- `품목`: 품목명·수량·단가·금액
- `원문_근거`: 처리 방식과 원문 근거

```text
course_outputs/receipt_result.xlsx
```

파일을 내려받아 실제 Excel에서 열고 세 시트가 있는지 확인합니다.

![JSON이 검증된 Excel로 바뀌는 과정](assets/07/02_json_to_excel.svg)

## 내 자료로 다시 할 때

내 문서는 정답을 자동으로 적용하면 안 됩니다. 원본을 직접 확인한 뒤 다음 정보를
남깁니다.

- 검토자
- 수정한 값
- 수정 이유
- 승인 또는 수정 후 승인

## 이번 시간의 정리

자동 검증은 계산과 형식을 확인하고, 사람은 원본과 업무 의미를 확인합니다. 두
조건이 모두 끝나야 Excel 다운로드를 허용합니다.

## 완료 확인

- 필수값과 합계 검증 결과를 확인했습니다.
- 승인 전 저장 차단을 확인했습니다.
- 세 시트가 있는 Excel을 만들었습니다.
- `course_outputs/receipt_result.xlsx`를 내려받았습니다.

다음 교시에는 영수증이 아닌 실제 업무 문서 사진에 VLM을 적용합니다.

## 참고자료

- [openpyxl 공식 문서](https://openpyxl.readthedocs.io/)
- [OWASP 스프레드시트 수식 주입 설명](https://owasp.org/www-community/attacks/CSV_Injection)
