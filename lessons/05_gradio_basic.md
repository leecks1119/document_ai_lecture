# 5교시. Python 함수에 화면 붙이기

> 합성 영수증 입력과 mock 결과를 연결한 가장 작은 Gradio 화면을 만듭니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/05_gradio_basic.ipynb)

## 1. 학습 목표

- Gradio의 입력·이벤트·출력 관계를 설명할 수 있다.
- 버튼과 Python 함수를 연결할 수 있다.
- 화면과 처리 함수를 분리해 확인할 수 있다.

## 2. 이번 교시의 결과물

- `app_05.py`: 파일 입력, OCR·VLM 선택, 인식 결과, JSON이 있는 기본 화면

## 3. 시작하기 전에

### 선수 지식

- 인자를 받고 값을 반환하는 Python 함수의 모양을 알고 있으면 충분하다.

### 준비 파일

- [합성 영수증](../sample_docs/receipt_sample.png)
- [mock OCR 텍스트](../sample_outputs/ocr_result.txt)
- [mock JSON](../sample_outputs/extracted_result.json)
- [5교시 Colab 노트북](../colab/05_gradio_basic.ipynb)

Colab에서 만든 Gradio 공유 주소는 공개될 수 있다. 합성 문서만 사용한다.

## 4. 핵심 개념

### 4.1 컴포넌트는 함수의 입출력을 화면에 보여 준다

- `File`: 문서 입력
- `Button`: 처리 시작 이벤트
- `Radio`: PaddleOCR 또는 PaddleOCR-VL 선택
- `Textbox`: 문서 인식 중간 결과
- `JSON`: 구조화 결과

![Gradio 입력과 버튼이 Python 함수를 거쳐 텍스트와 JSON으로 연결되는 흐름](assets/05/01_component_flow.svg)

### 4.2 버튼은 함수를 호출한다

`button.click()`에 함수, 입력 컴포넌트, 출력 컴포넌트를 순서대로 연결한다.

```python
process_button.click(
    fn=show_mock_result,
    inputs=[file_input, processor],
    outputs=[status, ocr_output, json_output],
)
```

### 4.3 화면과 처리 함수는 분리한다

> **쉬운 비유**
> Gradio 화면은 리모컨이고 Python 함수는 실제로 동작하는 기계다.

비유의 한계: 버튼이 보기 좋다고 처리 함수의 결과가 정확하거나 안전해지는 것은 아니다. 함수를 셀에서 직접 실행해 반환값을 먼저 확인한다.

![파일 입력, 처리 버튼, OCR 텍스트, JSON으로 구성된 최소 화면](assets/05/02_minimal_ui.svg)

## 5. 전체 실습 흐름

```text
show_mock_result() 직접 실행
  → 입력·출력 값 확인
  → Gradio 화면 골격 실행
  → 버튼과 함수 연결
  → app_05.py 저장
```

## 6. 단계별 실습

### 실습 1. 버튼과 mock 함수 연결 확인하기

처리 함수는 완성 코드로 제공된다.

```python
def show_mock_result(file_path, processor="PaddleOCR"):
    status = f"MOCK {processor} 결과 — 업로드 문서를 읽지 않았습니다."
    source = SAMPLE_VLM_MARKDOWN if processor == "PaddleOCR-VL" else SAMPLE_OCR_TEXT
    return status, source, SAMPLE_JSON
```

완성된 버튼 연결에서 함수·입력·출력 컴포넌트를 찾아본다.

```python
process_button.click(
    fn=show_mock_result,
    inputs=[file_input, processor],
    outputs=[status, ocr_output, json_output],
)
```

**기대 결과**

- 버튼을 누르면 상태에 `MOCK 결과`가 표시된다.
- OCR 텍스트와 합계가 `5000`인 JSON이 화면에 나타난다.

**mock 대체 경로**

Gradio 화면이 열리지 않으면 다음과 같이 함수를 직접 실행한다.

```python
show_mock_result(None)
```

같은 세 반환값이 나오면 이번 교시의 학습 목표는 달성한 것이다.

### 선택 실습. Colab 공유 주소 열기

공유 주소는 합성 샘플로만 확인하고 교육이 끝나면 세션을 종료한다. 운영 서비스로 사용하지 않는다.

## 7. Codex 활용

### 요청 목표

화면 골격에 버튼 연결 한 개만 추가한다.

### 실습 프롬프트

```text
목표: 제공된 Gradio Blocks 코드에서 처리 버튼을 show_mock_result에 연결해줘.
맥락: 입력은 File과 처리기 Radio, 출력은 상태·중간 결과·JSON 세 개야.
제약조건: CSS, 테마, 새 컴포넌트, 데이터베이스를 추가하지 마.
완료 기준: button.click 코드와 확인 방법만 알려줘.
```

### 생성 결과 확인

- 요청하지 않은 프레임워크를 추가하지 않았는가?
- 입력과 출력의 순서가 함수 반환값과 같은가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| `No module named gradio` | 설치 셀 미실행 | 노트북 설치 셀 실행 후 import 재시도 |
| 출력 순서가 바뀜 | 함수 반환값과 outputs 불일치 | 상태·텍스트·JSON 순서 맞추기 |
| 공유 주소가 열리지 않음 | 네트워크 또는 세션 종료 | 함수를 셀에서 직접 실행해 실습 계속 |

## 9. 형성평가

1. 버튼과 Python 함수를 연결할 때 필요한 세 요소는 무엇인가?
2. 화면이 열리면 추출값도 정확하다는 뜻인가?

<details>
<summary>정답 보기</summary>

1. 함수, 입력 컴포넌트, 출력 컴포넌트.
2. 아니다. 화면과 처리 결과는 별도로 검증한다.

</details>

## 10. 핵심 요약

- Gradio는 Python 함수의 입출력을 화면에 연결한다.
- 이벤트에는 함수·입력·출력 순서가 필요하다.
- 화면보다 처리 함수를 먼저 확인한다.

## 11. 완료 체크리스트

- [ ] `show_mock_result()`의 반환값을 확인했다.
- [ ] 버튼과 함수를 연결했다.
- [ ] `app_05.py`를 만들었다.

## 12. 다음 교시 예고

6교시에서는 업로드 확인, PaddleOCR·PaddleOCR-VL, JSON 변환을 작은 함수로 연결한다.

## 참고 자료

- [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks)
- [Gradio File](https://www.gradio.app/docs/gradio/file)
- [Gradio 앱 공유](https://www.gradio.app/guides/sharing-your-app)
