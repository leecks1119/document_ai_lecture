# 4교시 강사 노트

## 전달할 한 문장

오늘은 `PaddleOCR-VL-1.6-0.9B`를 실제로 실행하고, 모델의 문서 복원 결과와
Python 업무 규칙의 역할을 분리해 확인한다.

## 60분 운영

| 시간 | 행동 | 예상 증거 |
| ---: | --- | --- |
| 0~6 | 원본과 최종 산출물, 모델명 설명 | 이미지·Markdown·JSON |
| 6~12 | T4 GPU 확인과 설치 셀 실행 | GPU·파이프라인·모델명 |
| 12~27 | 5개 좌석 라인 순차 실제 추론 | `모델 실행 완료: True` |
| 27~34 | 실제 Markdown과 원시 JSON 관찰 | 표·읽기 순서·누락 |
| 34~42 | VLM Markdown→업무 JSON 변환 | 총액·품목·근거 |
| 42~49 | OCR+규칙 기준선과 비교 | 필드별 차이 |
| 49~55 | 원본 대조·`null` 확인 | 검토 행동 |
| 55~60 | 산출물과 provenance 확인 | 파일 5개 |

## 그대로 말하기

> “전체 파이프라인은 PaddleOCR-VL-1.6이고, 실제 비전-언어 모델은 그 안의 PaddleOCR-VL-1.6-0.9B입니다.”

> “지금 보이는 Markdown은 현재 이미지 픽셀을 모델에 넣어 만든 결과이고, 그다음 업무 JSON은 공개된 Python 규칙이 만든 결과입니다.”

> “VLM도 초안입니다. 원본에서 근거를 찾지 못하면 추측하지 말고 null로 둡니다.”

## 시연 중단 기준

공개 비식별 자료가 아니면 실행하지 않는다. 모델 다운로드가 진행 중인지는
Colab 출력으로 확인하고, GPU가 없으면 T4 런타임으로 다시 연결한다. 실제 추론이
실패한 상태를 준비 Markdown으로 성공 처리하지 않는다.

## 확인 질문과 정답

- JSON 문법이 맞으면 값도 맞나? → 아니오
- 날짜가 원문에 없으면? → `null`
- 실제 모델을 실행했음을 무엇으로 증명하나? →
  `model_executed=true`, 모델명, 입력 파일명, 실제 Markdown·원시 JSON
- VLM이 바로 영수증 업무 JSON을 만든 것인가? → 아니오. VLM은 문서를
  Markdown·구조 JSON으로 복원했고, Python 규칙이 업무 필드로 옮겼다.

## 종료 조건

`paddleocr_vl_raw.json`에서 `model_executed=true`를 확인하고
`paddleocr_vl_result.md`, `receipt_vlm.json`, `vlm_comparison.json`을 열어
실제 VLM 결과와 OCR+규칙 기준선의 총액 근거를 비교한다.
