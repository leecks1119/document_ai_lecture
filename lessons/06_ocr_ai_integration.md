# 6교시. 작은 함수들을 한 줄로 연결하기

> 업로드 확인, OCR, JSON 구조화를 하나의 처리 흐름으로 연결합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/06_ocr_ai_integration.ipynb)

## 1. 학습 목표

- 문서 처리 단계를 작은 함수로 나눌 수 있다.
- 단계별 상태와 오류를 사용자에게 표시할 수 있다.
- 실제 경로와 mock 경로를 혼동 없이 구분할 수 있다.

## 2. 이번 교시의 결과물

- `app_06.py`: 업로드 → OCR 또는 mock → JSON 결과가 연결된 미니 앱

## 3. 시작하기 전에

### 선수 지식

- 함수 호출과 `try/except`의 목적을 이해하면 충분하다.

### 준비 파일

- [합성 영수증](../sample_docs/receipt_sample.png)
- [6교시 Colab 노트북](../colab/06_ocr_ai_integration.ipynb)
- 저장소의 `src/pipeline.py`

## 4. 핵심 개념

### 4.1 작은 함수는 오류 위치를 보여 준다

```text
validate_upload()
  → extract_with_easyocr() 또는 load_mock_ocr()
  → mock_extract()
  → validate_receipt()
```

각 함수의 입력과 출력이 작으면 어느 단계에서 문제가 생겼는지 찾기 쉽다.

### 4.2 단계별 상태를 표시한다

성공 여부만 보여 주지 않고 어떤 모드를 사용했는지도 표시한다.

- `LIVE EasyOCR + MOCK 추출`
- `MOCK OCR + MOCK 추출`
- `입력 오류`
- `OCR 오류`

![LIVE 선택 경로와 mock 기본 경로의 상태 차이](assets/06/02_status_steps.svg)

### 4.3 mock은 사용자가 명시적으로 선택한다

> **쉬운 비유**
> 문서 처리는 바통을 넘기는 이어달리기다. 앞 단계가 넘긴 값을 다음 함수가 받는다.

비유의 한계: 실제 운영 시스템에는 재시도와 병렬 처리 등이 있지만 이번 과정에서는 한 줄 흐름만 다룬다.

업로드 실패 후 관련 없는 샘플 결과를 자동 표시하면 실제 결과로 오해할 수 있다. 오류를 먼저 보여 주고 사용자가 **샘플로 계속**을 선택한 뒤에만 mock을 실행한다.

![오류 표시 뒤 사용자가 샘플 경로를 선택하는 흐름](assets/06/01_live_mock_paths.svg)

## 5. 전체 실습 흐름

```text
process_document(use_sample=True) 확인
  → 업로드 오류 경로 확인
  → process_document() 안의 함수 연결
  → 화면에 모드·텍스트·JSON 표시
  → app_06.py 저장
```

## 6. 단계별 실습

### 실습 1. 통합 처리 함수 연결하기

보조 함수는 제공된다. 학습자는 처리 순서와 반환값을 확인한다.

```python
def process_document(file_path=None, *, use_sample=False):
    if use_sample:
        ocr_text = SAMPLE_OCR_TEXT
        status = "MOCK OCR + MOCK 추출"
    else:
        errors = validate_upload(file_path)
        if errors:
            return {"ok": False, "errors": errors}
        ocr_result = extract_with_easyocr(file_path)
        ocr_text = ocr_text_from_result(ocr_result)
        status = "LIVE EasyOCR + MOCK 추출"

    data = mock_extract(ocr_text)
    return {"ok": True, "status": status, "ocr_text": ocr_text, "data": data}
```

**기대 결과**

- `use_sample=True`에서 상태에 `MOCK`이 분명히 표시된다.
- 파일 없이 실제 경로를 실행하면 오류만 나오며 관련 없는 JSON은 표시되지 않는다.

**mock 대체 경로**

실제 EasyOCR는 선택 경로다. 필수 실습은 `process_document(use_sample=True)`로 끝까지 수행한다.

```python
mock_result = process_document(use_sample=True)
assert mock_result["ok"]
assert "MOCK" in mock_result["status"]
```

## 7. Codex 활용

### 요청 목표

자동 mock 전환이 없는지 통합 함수를 검토한다.

### 실습 프롬프트

```text
목표: process_document 함수의 오류와 mock 전환을 검토해줘.
맥락: 업로드 오류 뒤에는 사용자가 '샘플로 계속'을 눌러야 해.
제약조건: 오류를 숨기거나 자동으로 SAMPLE_JSON을 반환하지 마.
완료 기준: 실제 결과와 mock 결과를 혼동할 수 있는 코드만 찾아줘.
```

### 생성 결과 확인

- 오류 뒤에 자동으로 mock 데이터가 반환되지 않는가?
- 상태에 `LIVE` 또는 `MOCK`이 표시되는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 업로드 없이 오류 | 파일을 선택하지 않음 | 오류 확인 후 `샘플로 계속` 선택 |
| EasyOCR 오류 | 설치·모델 다운로드 실패 | 오류를 숨기지 말고 mock 버튼 사용 |
| JSON은 나오는데 상태가 없음 | 반환값 누락 | 결과에 `status` 포함 |

## 9. 형성평가

1. mock 사용 사실을 화면에 표시해야 하는 이유는 무엇인가?
2. 처리 단계를 작은 함수로 나누는 장점은 무엇인가?

<details>
<summary>정답 보기</summary>

1. 사용자가 실제 문서 처리 결과로 오해하지 않게 하기 위해서다.
2. 오류가 난 단계를 찾고 개별적으로 수정하기 쉽다.

</details>

## 10. 핵심 요약

- 통합은 작은 함수를 순서대로 연결하는 일이다.
- 단계별 상태와 오류를 숨기지 않는다.
- mock은 사용자의 명시적 선택 뒤에만 실행한다.

## 11. 완료 체크리스트

- [ ] 처리 단계의 함수 순서를 설명할 수 있다.
- [ ] 실제 경로와 mock 경로의 상태를 구분했다.
- [ ] `app_06.py`를 만들었다.

## 12. 다음 교시 예고

7교시에서는 JSON 결과의 필수값과 품목 합계를 확인하고 CSV로 저장한다.

## 참고 자료

- [Gradio File](https://www.gradio.app/docs/gradio/file)
- [Gradio 파일 접근 보안](https://www.gradio.app/guides/file-access)
- [EasyOCR Tutorial](https://www.jaided.ai/easyocr/tutorial/)
