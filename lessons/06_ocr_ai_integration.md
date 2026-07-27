# 6교시. OCR과 VLM 처리 경로 연결하기

> 문서 난이도에 따라 PaddleOCR 또는 PaddleOCR-VL을 고르는 흐름을 만듭니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/06_ocr_ai_integration.ipynb)

## 1. 학습 목표

- 단순 문서와 복잡한 문서에 맞는 처리 경로를 고를 수 있다.
- 처리 단계를 작은 함수로 연결할 수 있다.
- 실제 경로와 mock 경로의 상태·오류를 구분해 표시할 수 있다.

## 2. 이번 교시의 결과물

- `app_06.py`: OCR·VLM 선택과 JSON 변환을 연결한 미니 앱

## 3. 시작하기 전에

### 선수 지식

- 함수 호출과 `if` 조건문의 목적을 이해하면 충분하다.

### 준비 파일

- [합성 영수증](../sample_docs/receipt_sample.png)
- [6교시 Colab 노트북](../colab/06_ocr_ai_integration.ipynb)
- 저장소의 `src/pipeline.py`

## 4. 핵심 개념

### 4.1 쉬운 문서는 OCR부터 시작한다

- 글자 위치와 줄이 단순함: PaddleOCR
- 제목·표·여러 영역의 관계가 중요함: PaddleOCR-VL
- 비용과 오류 영향이 큼: 두 결과 비교와 사람 검토

모든 문서를 무조건 큰 VLM으로 처리하는 것이 정답은 아니다.

![단순 문서는 OCR, 복잡한 배치는 VLM, 중요한 값은 사람 검토로 이어지는 선택 지도](assets/06/02_status_steps.svg)

### 4.2 작은 함수는 오류 위치를 보여 준다

```text
validate_upload()
  → extract_with_paddleocr() 또는 parse_with_paddleocr_vl()
  → mock_extract()
  → validate_receipt()
```

각 단계의 입력과 출력이 작으면 파일·모델·변환 중 어디서 문제가 생겼는지 찾기 쉽다.

### 4.3 mock은 사용자가 명시적으로 선택한다

> **쉬운 비유**
> 문서 파이프라인은 환승 노선이다. OCR과 VLM은 다른 노선이고 상태 표시는 현재 탄 노선을 알려 주는 안내판이다.

비유의 한계: 실제 운영 시스템은 재시도·배치·관측 기능이 더 필요하다. 이번 과정은 한 문서의 한 줄 흐름만 다룬다.

![오류를 표시한 뒤 사용자가 샘플 경로를 선택하는 흐름](assets/06/01_live_mock_paths.svg)

업로드 실패 뒤 관련 없는 샘플을 자동 표시하지 않는다. 오류를 먼저 보여 주고 사용자가 **샘플로 계속**을 선택해야 한다.

## 5. 전체 실습 흐름

```text
processor에서 ocr 또는 vlm 선택
  → 업로드 오류 경로 확인
  → 명시적으로 sample 경로 실행
  → 상태·중간 결과·JSON 확인
  → app_06.py 저장
```

## 6. 단계별 실습

### 실습 1. 처리기 선택과 mock 경로 확인하기

```python
vlm_result = process_document(processor="vlm", use_sample=True)
ocr_result = process_document(processor="ocr", use_sample=True)

assert vlm_result["status"] == "MOCK PaddleOCR-VL + MOCK 추출"
assert ocr_result["status"] == "MOCK PaddleOCR + MOCK 추출"
```

실제 업로드 경로에서는 `processor`에 따라 공통 파이프라인의 OCR 또는 VLM 함수가 호출된다.

```python
live_result = process_document(
    "receipt_sample.png",
    processor="ocr",
    use_sample=False,
)
```

**기대 결과**

- 두 mock 상태에 어떤 처리기를 흉내 냈는지 표시된다.
- 파일 없이 실제 경로를 실행하면 오류만 반환하고 관련 없는 JSON은 표시하지 않는다.

**mock 대체 경로**

모델 설치가 어려우면 `use_sample=True`를 직접 선택한다. 오류가 난 뒤 자동으로 mock 결과를 반환하는 코드는 사용하지 않는다.

## 7. Codex 활용

### 요청 목표

OCR·VLM 선택과 mock 표시가 섞이지 않았는지 검토한다.

### 실습 프롬프트

```text
목표: process_document의 ocr, vlm, mock 분기를 검토해줘.
맥락: 모델 오류 뒤에는 사용자가 직접 샘플 경로를 선택해야 해.
제약조건: 오류를 숨기거나 자동으로 mock 결과를 반환하지 마.
완료 기준: 처리기 이름과 실제/mock 상태가 잘못 표시되는 경우만 알려줘.
```

### 생성 결과 확인

- OCR과 VLM 함수가 올바른 분기에서 호출되는가?
- 상태에 처리기와 `LIVE` 또는 `MOCK`이 표시되는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 파일 없이 오류 | 업로드하지 않음 | 오류 확인 후 `샘플로 계속` 선택 |
| PaddleOCR 설치 오류 | 런타임·모델 다운로드 문제 | 선택 실행 중지 후 mock 경로 사용 |
| VLM이 느림 | 큰 모델과 메모리 사용 | GPU 런타임 또는 OCR 경로 선택 |

## 9. 형성평가

1. 단순한 한 줄 영수증은 어느 경로부터 시작하는가?
2. mock 사실을 상태에 표시하는 이유는 무엇인가?

<details>
<summary>정답 보기</summary>

1. 비용이 작은 PaddleOCR 경로부터 시작한다.
2. 실제 업로드 처리 결과로 오해하지 않도록 하기 위해서다.

</details>

## 10. 핵심 요약

- 문서 구조에 따라 OCR·VLM·사람 검토를 고른다.
- 작은 함수를 연결하면 오류 위치를 찾기 쉽다.
- mock은 명시적으로 선택하고 상태에 표시한다.

## 11. 완료 체크리스트

- [ ] OCR과 VLM 선택 기준을 설명했다.
- [ ] 실제와 mock 상태를 구분했다.
- [ ] `app_06.py`를 만들었다.

## 12. 다음 교시 예고

7교시에서는 JSON의 필수값과 품목 합계를 검증한 뒤 CSV로 저장한다.

## 참고 자료

- [PaddleOCR 빠른 시작](https://www.paddleocr.ai/latest/en/quick_start.html)
- [PaddleOCR-VL 파이프라인](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
