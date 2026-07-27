# 교육 환경

## 기본 환경

| 항목 | 기준 |
| --- | --- |
| 실습 환경 | Google Colab |
| 검증 런타임 | `2026.04` 고정 런타임 우선 |
| Python | 3.12.x |
| 기본 하드웨어 | CPU |
| 웹앱 | Streamlit 1.60.0 |
| 기본 OCR 시도 | PaddleOCR 3.7.0 + PP-OCRv5 Korean |
| 강사 시연 문서 VLM | PaddleOCR-VL 1.6 또는 상용 VLM 1회 |
| 선택 PDF 변환 | PyMuPDF 1.28.0 |

Colab은 관리형 환경이므로 로컬 Python의 패치 버전과 항상 같지 않다. 수업 시작 셀에서 실제 버전을 출력하고, 문제가 있으면 **런타임 → 런타임 유형 변경**에서 강사가 안내한 고정 런타임을 선택한다.

근거: [Google Colab 런타임 버전 FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)

## 필수 경로

1~8교시 필수 경로에는 다음이 필요하지 않다.

- GPU
- 생성형 AI API 키
- Google Drive 연결
- 학습자 결제

2교시는 식별정보를 가린 실제 영수증 한 장에 OCR을 시도한다. 실행이 3분 안에 끝나지 않으면 노트북에 포함된 준비 결과로 전환해 같은 원본 대조·구조화·검증 과정을 계속한다.

## 기본 PaddleOCR와 3분 복구 기준

수업 전날과 당일에 20대 PC의 패키지·모델 캐시와 한국어 영수증 실행을 확인한다. 설치·다운로드·첫 실행이 3분을 넘으면 중지하고 `MOCK_OCR_RESULT`로 전환한다. 준비 결과 사용 여부를 화면과 산출물에 표시한다.

## 선택 PaddleOCR-VL

4교시 선택 셀의 기본값은 다음과 같다.

```python
RUN_PADDLEOCR_VL = False
```

문서 전용 멀티모달 모델은 OCR보다 다운로드와 메모리 사용량이 크다. 실제 호출은 강사가 비식별 샘플로 한 번 시연하고, 학습자는 준비된 VLM Markdown으로 실습한다.

## Streamlit과 Colab

모든 학습자 실습은 Colab에서 진행한다. 5교시는 `app_05.py`를 만든 뒤 Streamlit AppTest로 화면 코드를 검사한다.

- 화면 실습에는 비식별 공개 샘플이나 합성 문서만 사용한다.
- 5MB 이하 PNG·JPEG·PDF만 사용한다.
- 개인 영수증은 공개 웹앱 주소나 외부 API에 업로드하지 않는다.
- 필수 실습은 공개 터널·로컬 PC·별도 서버를 요구하지 않는다.
- 화면이 열리지 않아도 AppTest와 처리 함수 호출로 같은 결과를 확인할 수 있다.

## 개발자용 로컬 검증 환경

로컬 검증이 필요하면 Python 3.12 가상환경을 사용한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

실제 PaddleOCR까지 확인할 때만 다음을 설치한다.

```bash
python -m pip install -r requirements-ocr.txt
```

PaddleOCR-VL까지 확인하려면 다음을 설치한다.

```bash
python -m pip install -r requirements-vlm.txt
```

학습자는 이 로컬 환경을 구성하지 않는다. 저장소 유지보수와 강사 사전 검증에만 사용한다.

## 수업 전 확인 명령

```bash
python -m compileall .
pytest
python tools/validate_colab_notebooks.py
```
