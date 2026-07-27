# 5교시. 문서 자동화 웹 애플리케이션 기본 구현

> 1~4교시에서 확인한 Python 처리 결과에 파일 입력·실행 버튼·결과 화면을 붙여 작은 웹앱 코드로 만듭니다.
>
> **핵심 메시지:** Streamlit은 Python 처리 함수를 사용자가 실행하고 결과를 확인할 수 있는 웹 화면으로 바꾸는 도구입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/05_streamlit_basic.ipynb)

## 1. 학습 목표

- Streamlit 코드에서 입력·실행·출력 부분을 찾을 수 있다.
- 파일 입력과 준비된 처리 결과를 한 화면에 연결할 수 있다.
- Colab에서 실제 서버를 공개하지 않고 `AppTest`로 화면 코드를 검사할 수 있다.

## 2. 이번 교시의 결과물

- `app_05.py`: 한 장의 문서 입력과 판독 원문·JSON 결과 영역을 가진 Streamlit 앱

## 3. 시작하기 전에

### 선수 지식

- Python 함수를 호출하면 결과가 돌아온다는 뜻을 이해하면 충분하다.

### 준비 파일

- 4교시의 `receipt.json` 또는 노트북의 완성 복구본
- [비식별 한국 실물 영수증](../sample_docs/public_receipts/korea/taebaek_restaurant_2025_redacted.png)
- [5교시 Colab 노트북](../colab/05_streamlit_basic.ipynb)

모든 실습은 Colab에서 진행한다. 개인 영수증 대신 비식별 공개 샘플을 사용하며, 외부 공개 터널이나 배포 주소를 만들지 않는다.

## 4. 핵심 개념

### 4.1 Streamlit 앱도 위에서 아래로 실행되는 Python 파일이다

```python
import streamlit as st

st.title("영수증 Document AI 미니 앱")
uploaded = st.file_uploader("영수증 한 장")
run = st.button("처리")
```

`st.title()`은 제목, `st.file_uploader()`는 파일 입력, `st.button()`은 실행 시점을 만든다.

### 4.2 화면과 문서 처리 함수의 역할은 다르다

```text
Streamlit 화면: 파일을 받고 버튼 클릭을 전달한다.
처리 함수: OCR·구조화·검증 결과를 반환한다.
Streamlit 화면: 반환된 원문·JSON·오류를 보여 준다.
```

화면이 정상적으로 열려도 추출값이 정확하다는 뜻은 아니다. 결과의 정확성은 원문 대조와 검증 규칙으로 따로 확인한다.

### 4.3 Colab에서는 AppTest로 웹앱 코드를 검사한다

Streamlit의 `AppTest`는 브라우저 서버를 열지 않고도 제목·파일 입력·버튼·오류를 Python 코드로 확인한다. 따라서 수업에서는 공개 링크 없이도 웹앱 구현 여부를 검증할 수 있다.

## 5. 전체 실습 흐름

```text
Colab에서 app_05.py 작성
  → 파일 입력·버튼·결과 영역 연결
  → Streamlit AppTest 실행
  → 예외가 없는지 확인
  → app_05.py 다운로드
```

## 6. 단계별 실습

### 실습 1. 준비된 처리 결과를 화면에 연결하기

노트북의 시작 코드에서 빈칸 세 곳만 채운다.

```python
import streamlit as st

st.title("영수증 Document AI 미니 앱")
uploaded = st.file_uploader(
    "영수증 이미지 또는 PDF 한 장",
    type=["png", "jpg", "jpeg", "pdf"],
)

if st.button("준비 결과로 실행"):
    st.info("준비 결과를 사용했습니다.")
    st.text_area("판독 원문", SAMPLE_TEXT)
    st.json(SAMPLE_JSON)
```

Colab에서 파일을 저장한 뒤 화면 코드를 검사한다.

```python
from streamlit.testing.v1 import AppTest

app_test = AppTest.from_file("app_05.py").run(timeout=20)
assert not app_test.exception
assert len(app_test.file_uploader) == 1
assert len(app_test.button) == 1
```

**기대 결과**

- `app_05.py`가 생성된다.
- AppTest 결과에 예외가 없다.
- 파일 입력 한 개와 실행 버튼 한 개가 확인된다.

**완성 복구본**

빈칸 수정이 어려우면 노트북의 전체 정답 `app_05.py`를 저장하고 AppTest 결과를 확인한다. 실제 웹 서버를 열지 못해도 같은 산출물을 만든다.

## 7. 실습 결과 확인

- 파일을 여러 장이 아니라 한 장만 받는가?
- 실행 버튼을 누르기 전에는 준비 결과를 실제 처리처럼 표시하지 않는가?
- 판독 원문과 구조화 JSON을 서로 다른 영역에 보여 주는가?
- Colab AppTest에서 예외가 없는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| `No module named streamlit` | 설치 셀 미실행 | `streamlit==1.60.0` 설치 셀 실행 |
| AppTest에 예외 표시 | 변수명 또는 들여쓰기 오류 | 전체 정답의 해당 줄과 비교 |
| 버튼을 눌러도 결과 없음 | `if st.button(...)` 아래 코드 누락 | 준비 결과 표시 세 줄 확인 |
| 실제 화면을 열 수 없음 | Colab에서 서버를 공개하지 않음 | AppTest로 필수 실습 완료 |

## 9. 형성평가

1. Streamlit의 역할은 OCR 모델을 만드는 것인가, Python 처리 함수를 화면에 연결하는 것인가?
2. AppTest가 통과하면 추출값의 정확성도 보장되는가?

<details>
<summary>정답 보기</summary>

1. Python 처리 함수를 화면에 연결하는 것이다.
2. 아니다. UI 코드와 추출값 검증은 별개의 확인이다.

</details>

## 10. 핵심 요약

- Streamlit은 Python 스크립트를 웹앱 화면으로 표현한다.
- 파일 입력·실행 버튼·결과 영역이 처리 함수와 연결돼야 한다.
- 모든 필수 구현과 검사는 Colab에서 수행한다.
- 화면 동작과 문서 데이터 정확성은 따로 검증한다.

## 11. 완료 체크리스트

- [ ] Colab에서 `app_05.py`를 만들었다.
- [ ] 파일 입력과 버튼을 연결했다.
- [ ] AppTest에서 예외가 없는지 확인했다.
- [ ] `app_05.py`를 다운로드했다.

## 12. 다음 교시 예고

6교시에서는 준비 결과 대신 실제 OCR 또는 복구 결과와 JSON 추출 함수를 Streamlit 앱에 연결한다.

## 참고 자료

- [Streamlit 시작하기](https://docs.streamlit.io/get-started)
- [Streamlit `st.file_uploader`](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
