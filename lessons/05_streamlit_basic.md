# 5교시. 문서 자동화 웹 애플리케이션 기본 구현

> **이번 교시의 한 문장:** 파일 입력·실행 버튼·결과 화면을 처리 함수에 붙이면 사람이 사용할 수 있는 도구의 형태가 됩니다.

[Colab 실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/05_streamlit_basic.ipynb)

## 60분 뒤 남길 것

- 업로드, 버튼, 원문, JSON 영역을 가진 Streamlit 앱을 만듭니다.
- 업로드한 파일명이 실제 화면에 반영되는지 확인합니다.
- 브라우저를 공개하지 않고 AppTest로 동작을 검사합니다.
- `course_outputs/app_05.py`를 만듭니다.

## 개념 10%: 화면과 처리 함수는 역할이 다릅니다

```text
화면: 파일을 받고 상태와 결과를 보여 줌
처리 함수: 파일을 읽고 OCR·구조화·검증을 수행
```

파일을 올렸는데 파일명·바이트 수·결과가 바뀌지 않는다면 업로드가 실제 처리와 연결된 것이 아닙니다.

![업로드, 처리 함수, 화면 출력으로 이어지는 최소 앱 구조](assets/05/01_component_flow.svg)

![파일 선택, 실행 버튼, 결과 영역으로 이루어진 최소 화면](assets/05/02_minimal_ui.svg)

## 실습 90%

### 1. 앱 코드의 네 부분을 찾습니다

```python
st.title(...)
uploaded = st.file_uploader(...)
st.button(...)
st.text_area(...)
st.json(...)
```

### 2. 업로드가 장식이 아닌지 확인합니다

파일을 선택하면 다음 값이 달라져야 합니다.

```text
업로드 연결 확인: 파일명 · 바이트 수
```

5교시는 화면 연결을 다루므로 OCR 실행은 6교시에서 붙입니다.

### 3. 공개 샘플 준비 결과를 표시합니다

버튼을 누르면 화면에 다음 세 가지가 나타납니다.

- `실행 모드: PREPARED_FALLBACK`
- 공개 한국 영수증의 판독 원문
- 상호명·날짜·품목·총액 JSON

### 4. AppTest로 검사합니다

Colab에서는 공개 Streamlit 터널이나 배포 주소를 만들지 않습니다. `streamlit.testing.v1.AppTest`로 다음을 확인합니다.

```text
제목 1개
파일 업로더 1개
버튼 1개
버튼 클릭 후 PREPARED_FALLBACK 표시
```

정상 결과:

```text
CHECKPOINT 1/1 PASS: 업로드·버튼·결과 화면
```

아래는 같은 준비 결과 경로를 실제 앱에서 실행한 화면입니다. 상태 문구와 결과 영역이 함께 보이는지 비교합니다.

![공개 한국 영수증 준비 결과를 표시한 실제 Streamlit 앱](assets/screens/app_prepared_result.png)

## 통과 기준

- `app_05.py`가 생성되었습니다.
- 업로드된 파일의 이름과 크기를 코드가 실제로 읽습니다.
- 준비 결과 버튼을 누른 뒤 원문과 JSON이 표시됩니다.
- 외부 공개 주소 없이 AppTest가 통과합니다.

## 막혔을 때

- `ModuleNotFoundError: streamlit`이면 설치 셀부터 다시 실행합니다.
- AppTest에 제목이 없으면 앱 코드가 저장된 뒤 검사 셀을 실행했는지 확인합니다.
- 파일을 업로드했는데 메시지가 변하지 않으면 `uploaded.getvalue()` 연결을 확인합니다.

다음 교시에는 업로드한 파일을 실제 PaddleOCR 함수에 연결하고 실패·복구 상태를 화면에 표시합니다.

## 참고 자료

공식 근거는 [과정 참고자료와 적용 범위](../docs/course_references.md)의 5교시 표에서 확인할 수 있습니다.
