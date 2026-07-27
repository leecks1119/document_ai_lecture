# 6교시. OCR 및 정보 추출 기능 연동

> 파일에서 OCR 또는 준비 텍스트를 얻고, 추출 함수에 연결해 영수증 한 장이 JSON이 되는 한 줄 흐름을 완성합니다.
>
> **핵심 메시지:** 입력 형식과 업무 위험에 맞는 가장 단순한 처리 경로를 고르고, 실제 처리와 준비 결과 사용 여부를 숨기지 않아야 합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/06_ocr_ai_integration.ipynb)

## 1. 학습 목표

- `파일 → OCR 또는 준비 텍스트 → 추출 함수 → JSON` 흐름을 연결할 수 있다.
- 입력 형식에 따라 네이티브 파서·PDF 텍스트층·OCR·VLM 중 첫 경로를 고를 수 있다.
- 실제 경로와 준비 결과 경로의 상태·오류를 구분해 표시할 수 있다.

## 2. 이번 교시의 결과물

- `app_06.py`: 영수증 한 장의 판독 결과를 JSON에 연결하는 Streamlit 앱

## 3. 시작하기 전에

### 선수 지식

- 함수 호출과 `if` 조건문의 목적을 이해하면 충분하다.

### 준비 파일

- 5교시의 기본 Streamlit 앱 또는 교재에 포함된 완성 복구본
- [합성 영수증](../sample_docs/receipt_sample.png)
- [6교시 Colab 노트북](../colab/06_ocr_ai_integration.ipynb)
- 노트북에 포함된 독립 실행 완성 복구본

이번 시간에는 사진 영수증의 OCR 결과 또는 준비 텍스트를 JSON 함수에 연결하는 한 경로만 직접 완성합니다. Excel·Word·PPT 원본, 텍스트 PDF, 표 캡처와 사진의 처리 차이는 준비된 선택 예시에서 확인합니다.

`app_06.py`에는 준비 데이터, 처리 함수, Streamlit 화면을 함께 저장합니다. 파일을 저장한 뒤 AppTest에서 오류 없이 열리는지 확인합니다.

## 4. 핵심 개념

### 4.1 쉬운 문서는 OCR부터 시작합니다

- Excel·Word·PPT 원본: 네이티브 구조 파서 우선
- 텍스트가 선택되는 PDF: 텍스트층 우선
- 표 캡처·사진·스캔: PaddleOCR 우선
- 제목·표·여러 영역의 시각 관계가 중요함: 문서 VLM 검토
- 오류 영향이 큰 값: 원문 대조와 사람 확인

모든 문서를 이미지로 바꾸거나 무조건 큰 VLM으로 처리하는 것이 정답은 아닙니다.

![단순 문서는 OCR, 복잡한 배치는 VLM, 중요한 값은 사람 검토로 이어지는 선택 지도](assets/06/02_status_steps.svg)

### 4.2 작은 함수는 오류 위치를 보여 줍니다

```text
validate_upload()
  → extract_with_paddleocr() 또는 parse_with_paddleocr_vl()
  → extract_prepared_result()
  → validate_receipt()
```

각 단계의 입력과 출력이 작으면 파일·모델·변환 중 어디서 문제가 생겼는지 찾기 쉽습니다.

### 4.3 준비 결과는 직접 선택합니다

> **쉬운 비유**
> 문서 파이프라인은 환승 노선과 같습니다. OCR과 VLM은 다른 노선이고 상태 표시는 현재 탄 노선을 알려 주는 안내판입니다.

비유의 한계: 이번 과정은 한 문서의 한 줄 흐름만 다룹니다. 실제 업무 시스템에는 추가 운영 기능이 필요할 수 있습니다.

![오류를 표시한 뒤 사용자가 샘플 경로를 선택하는 흐름](assets/06/01_live_mock_paths.svg)

업로드 처리에 실패하면 먼저 오류를 확인합니다. 계속하려면 직접 **샘플로 계속**을 선택합니다.

## 5. 전체 실습 흐름

```text
파일 형식과 사용 가능한 원본 확인
  → 사진 영수증에서 OCR 또는 준비 텍스트 선택
  → 추출 함수에 연결
  → 상태·중간 결과·JSON 확인
  → app_06.py 저장
  → AppTest에서 파일 입력·두 버튼·준비 결과 확인
```

## 6. 단계별 실습

### 실습 1. 처리기 선택과 준비 결과 경로 확인하기

```python
vlm_result = process_document(processor="vlm", use_sample=True)
ocr_result = process_document(processor="ocr", use_sample=True)

assert vlm_result["status"] == "준비 결과: PaddleOCR-VL + 추출"
assert ocr_result["status"] == "준비 결과: PaddleOCR + 추출"
```

실제 업로드 경로에서는 `processor`에 따라 공통 파이프라인의 OCR 또는 VLM 함수가 호출됩니다.

```python
live_result = process_document(
    "receipt_sample.png",
    processor="ocr",
    use_sample=False,
)
```

**기대 결과**

- 두 준비 상태에 어떤 처리기를 대신 사용했는지 표시됩니다.
- 파일 없이 실제 경로를 실행하면 오류만 반환하고 관련 없는 JSON은 표시하지 않습니다.
- `app_06.py`가 다른 저장소 모듈 없이 import되고 AppTest에서 예외가 없습니다.

**준비 결과 경로**

모델 설치가 어렵다면 `use_sample=True`를 직접 선택합니다. 전체 시도가 3분을 넘으면 실행을 중지하고 필요한 셀을 위에서 아래로 다시 실행합니다. 오류가 난 뒤 자동으로 준비 결과를 반환하는 코드는 사용하지 않습니다.

## 7. 실습 결과 확인

- 파일에서 판독 원문을 얻은 뒤 JSON 추출 함수가 실행되는가?
- 실제 처리와 준비 결과의 상태가 화면에 구분되어 보이는가?
- 오류가 발생했을 때 관련 없는 샘플 결과로 조용히 바뀌지 않는가?
- AppTest에서 파일 입력 한 개와 `업로드 처리`·`샘플로 계속` 버튼 두 개가 확인되는가?
- 통합 코드를 Colab에서 실행하고 내려받았는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 파일 없이 오류 | 업로드하지 않음 | 오류 확인 후 `샘플로 계속` 선택 |
| PaddleOCR 실행 오류 | 환경 준비 또는 다운로드 문제 | 실행 중지 후 준비 결과 선택 |
| 준비 결과도 실행되지 않음 | 셀 실행 순서 문제 | 필요한 셀을 위에서 아래 재실행 |
| 파일 저장 오류 | 이전 파일과 이름 충돌 | 전체 정답 `app_06.py`를 다시 저장 |
| AppTest 예외 | 들여쓰기·변수 오류 | 전체 정답과 해당 줄 비교 |

## 9. 형성평가

1. 단순한 한 줄 영수증은 어느 경로부터 시작하는가?
2. 준비 결과 사용 사실을 상태에 표시하는 이유는 무엇인가?

<details>
<summary>정답 보기</summary>

1. 단순한 사진 영수증은 먼저 OCR 경로를 검토합니다.
2. 실제 업로드 처리 결과로 오해하지 않도록 하기 위해서입니다.

</details>

## 10. 핵심 요약

- 문서 구조에 따라 OCR·VLM·사람 검토를 고릅니다.
- 작은 함수를 연결하면 오류 위치를 찾기 쉽습니다.
- 준비 결과는 직접 선택하고 상태에 표시합니다.

## 11. 완료 체크리스트

- [ ] OCR과 VLM 선택 기준을 설명했다.
- [ ] 실제와 준비 결과 상태를 구분했다.
- [ ] `app_06.py`를 만들었다.
- [ ] Colab AppTest에서 독립 실행을 확인했다.

## 12. 다음 교시 예고

7교시에서는 JSON의 필수값과 품목 합계를 검증한 뒤 `receipt_result.xlsx`로 저장합니다.

## 참고 자료

- [PaddleOCR 빠른 시작](https://www.paddleocr.ai/latest/en/quick_start.html)
- [PaddleOCR-VL 파이프라인](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [과정 참고자료와 적용 범위](../docs/course_references.md)
