# 강사용 환경·모델 운영

> 수강생에게는 Colab 실행 셀과 짧은 복구 행동만 보여 준다. 이 문서는 수업 전 검증과 시연 판단에 사용한다.

## 기준 환경

| 항목 | 강사용 기준 |
| --- | --- |
| Colab | `2026.04` 고정 런타임 우선, 실제 Python 버전은 첫 셀에서 확인 |
| Python | 3.12.x |
| OCR | PaddleOCR 3.7.0 + PP-OCRv5 Korean |
| 문서 VLM | PaddleOCR-VL-1.6 전체 파이프라인 + PaddleOCR-VL-1.6-0.9B 실제 모델 |
| 웹앱 | Streamlit 1.60.0, Colab 내부 iframe 미리보기 |
| 기본 비용 | 수강생 API 키·결제 없음 |

정확한 버전은 재현을 위한 고정값이다. 새 버전을 쓰려면 한국어 지원,
노트북 실행, 실제 모델 결과, 웹앱 미리보기를 모두 다시 검증한다.

## 수업 전날 확인

1. Colab 링크 8개를 새 세션에서 연다.
2. `python tools/validate_colab_notebooks.py`를 실행한다.
3. PP-OCRv5 Korean 설치·모델 다운로드·한국 영수증 인식을 확인한다.
4. 실제 44토큰 회귀 사례가 합계 76,000원·품목 5개를 복원하는지 확인한다.
5. 5·6교시 iframe 미리보기와 7교시 잘못된 총액의 Excel 차단을 확인한다.
6. 4·8교시 `model_executed=true`, 실제 Markdown, 원시 JSON 생성을 확인한다.
7. 자동 검증용 기록 결과는 수업 흐름 검증에만 사용하고 실제 모델 실행
   증거로 쓰지 않는다.

## 실제 OCR 운영

- 2·3·6교시에서 실제 PP-OCRv5를 실행한다.
- 첫 모델 다운로드가 길어질 수 있으므로 2교시에서 설치를 끝내고 같은
  런타임을 유지한다.
- 3·6교시 결과는 현재 선택한 사진에서 생성됐는지 파일명과 원본으로 확인한다.
- 한 명의 설치 문제로 전체 진행을 멈추지 않는다.

## 문서 VLM 실습

- 수강생은 API 키를 입력하지 않는다.
- 모든 수강생은 4교시에서 공개 비식별 샘플 한 장을 실제 실행한다.
- Colab에서 T4 GPU를 선택하고 5개 좌석 라인을 1분 간격으로 시작한다.
- 실행 모델은 `PaddleOCR-VL-1.6-0.9B`, 전체 파이프라인은
  `PaddleOCR-VL-1.6`, 로컬 추론 엔진은 `transformers`다.
- GPU·메모리·다운로드 오류를 준비된 Markdown으로 성공 처리하지 않는다.
- 결과가 곧 정답이라는 표현을 피하고 원문 근거와 `null` 원칙을 바로 연결한다.

## 로컬 검증

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall .
pytest
python tools/validate_course_materials.py
python tools/validate_colab_notebooks.py
python tools/validate_office_samples.py
```

OCR·VLM 실제 실행 검증이 필요할 때만 각각 `requirements-ocr.txt`, `requirements-vlm.txt`를 추가로 설치한다.
