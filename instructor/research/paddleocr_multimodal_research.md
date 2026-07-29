# 2026 PaddleOCR·멀티모달 기술 정정

> 강사·교재 유지보수자 전용 기술 결정 기록

조사 기준: 2026-07-27  
목적: EasyOCR 중심 설계를 최신 Document AI 입문 과정으로 교체

## 결론

기존 과정의 EasyOCR 기본 선택은 폐기한다. 2026 입문 과정은 다음 두 경로와
하이브리드 사용 원칙을 가르친다.

| 경로 | 교육용 기준 | 역할 |
| --- | --- | --- |
| OCR | PaddleOCR 3.7.0 + PP-OCRv5 Korean | 텍스트·좌표·신뢰도 추출 |
| 멀티모달 | PaddleOCR-VL 1.6 | 레이아웃·표·읽기 순서를 포함한 문서 파싱 |
| 하이브리드 | 두 결과 + 업무 스키마 검증 | 근거 확인과 최종 JSON 확정 |

## 왜 PP-OCRv5 Korean인가

PaddleOCR 3.7.0의 일반 OCR 기본 모델은 PP-OCRv6다. 그러나 PP-OCRv6의
현재 통합 언어 목록에는 한국어가 포함되지 않는다. 한국어 합성 영수증은
`lang="korean"`, `ocr_version="PP-OCRv5"`를 명시한다.

이는 최신 패키지에서 구형 API를 쓰는 것이 아니라, 최신 PaddleOCR 3.7
파이프라인에서 한국어를 지원하는 모델 계열을 명시적으로 선택하는 것이다.

## 왜 PaddleOCR-VL 1.6인가

PaddleOCR-VL은 레이아웃 분석 뒤 각 영역을 VLM으로 인식하고 읽기 순서대로
합치는 문서 전용 멀티모달 파이프라인이다. VLM 구성요소만 단독 실행하는 것과
완전한 PaddleOCR-VL 파이프라인은 다르므로 교재에서도 이를 구분한다.

PaddleOCR-VL 1.6은 2026-05-28 공개됐고, 기본 파이프라인 버전은 `v1.6`이다.
결과에서 `parsing_res_list`, Markdown, JSON을 얻을 수 있다.

## Colab 실행 기준

### 실제 OCR 선택 셀

```python
RUN_PADDLEOCR = False
```

`True`일 때만 PaddlePaddle과 PaddleOCR 3.7.0을 설치하고 합성 영수증을
처리한다. 최초 모델 다운로드 실패가 수업 전체를 막지 않도록 mock OCR 결과를
기본 경로로 유지한다.

### 실제 멀티모달 필수 셀

```python
pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    engine="transformers",
)
page_results = list(pipeline.predict(str(VLM_INPUT_PATH)))
```

4교시 필수 경로는 현재 이미지 픽셀을 실제 모델에 전달한다. 빠른 저장소
자동검사만 모델 다운로드를 생략하며, 이 경우 `model_executed=false`와
VLM 값 `null`을 유지한다. 준비된 Markdown을 실제 추론 결과로 대체하지 않는다.

## 초보자 범위

- 교시당 핵심 개념 3개, 기본 실습 1개, 주 산출물 1개를 유지한다.
- PaddleOCR와 PaddleOCR-VL의 내부 모델 구조는 한 문단 이상 설명하지 않는다.
- PP-StructureV3, 파인튜닝, 추론 서버, Docker, 벤치마크 비교는 필수 범위에서 제외한다.
- “VLM이 문서를 완전히 이해한다”거나 “멀티모달이면 검증이 필요 없다”고 설명하지 않는다.
- 실제 개인정보는 로컬 모델·Colab·외부 API 어디에도 입력하지 않는다.

## 공식 근거

- [PaddleOCR 3.7.0 배포 정보](https://pypi.org/project/paddleocr/)
- [PaddleOCR 3.x Quick Start](https://www.paddleocr.ai/latest/en/quick_start.html)
- [일반 OCR 파이프라인](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [PP-OCRv5 다국어·한국어](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
- [PaddleOCR-VL 1.6](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)
- [PaddleOCR-VL 사용법](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleOCR 공식 API Python SDK](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/serving/paddleocr_official_api/python.html)
