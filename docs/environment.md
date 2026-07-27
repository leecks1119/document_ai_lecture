# 교육 환경

## 기본 환경

| 항목 | 기준 |
| --- | --- |
| 실습 환경 | Google Colab |
| 검증 런타임 | `2026.04` 고정 런타임 우선 |
| Python | 3.12.x |
| 기본 하드웨어 | CPU |
| UI | Gradio 6.20.0 |
| 선택 OCR | EasyOCR 1.7.2 |
| 선택 PDF 변환 | PyMuPDF 1.28.0 |

Colab은 관리형 환경이므로 로컬 Python의 패치 버전과 항상 같지 않다. 수업 시작 셀에서 실제 버전을 출력하고, 문제가 있으면 **런타임 → 런타임 유형 변경**에서 강사가 안내한 고정 런타임을 선택한다.

근거: [Google Colab 런타임 버전 FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)

## 필수 경로

1~8교시 기본 경로에는 다음이 필요하지 않다.

- GPU
- 실제 OCR 모델 다운로드
- 생성형 AI API 키
- Google Drive 연결
- 실제 업무 문서

노트북에 포함된 합성 영수증, OCR 결과, JSON으로 모든 산출물을 만든다.

## 선택 EasyOCR

2교시 선택 셀의 기본값은 다음과 같다.

```python
RUN_OPTIONAL_EASYOCR = False
```

강사가 모델 다운로드 가능 여부를 확인한 경우에만 `True`로 바꾼다. 사내망이나 교육장 네트워크에서 실패하면 즉시 `False`로 되돌리고 mock 결과를 사용한다.

## 생성형 AI API 연결 전 준비 확인

4교시 준비 확인 셀의 기본값은 다음과 같다.

```python
CHECK_OPTIONAL_API_READINESS = False
```

이번 입문 과정은 실제 API를 호출하지 않는다. Secrets 같은 비밀 저장 방식, 조직 승인, 데이터 보존·학습 이용·리전·계약 조건을 확인한 뒤 별도 심화 과정에서 연결한다. 실제 개인정보 문서는 사용하지 않는다.

## Gradio와 공개 주소

Colab에서 `share=True`로 실행하면 외부에서 접근 가능한 주소가 만들어질 수 있다.

- 합성 문서만 업로드한다.
- 5MB 이하 PNG·JPEG·PDF만 사용한다.
- 교육이 끝나면 런타임을 종료한다.
- 공유 주소를 운영 서비스로 사용하지 않는다.
- 화면이 열리지 않아도 함수를 직접 호출해 같은 결과를 확인할 수 있다.

## 로컬 선택 환경

로컬 검증이 필요하면 Python 3.12 가상환경을 사용한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

실제 EasyOCR까지 확인할 때만 다음을 설치한다.

```bash
python -m pip install -r requirements-ocr.txt
```

## 수업 전 확인 명령

```bash
python -m compileall .
pytest
python tools/validate_colab_notebooks.py
```
