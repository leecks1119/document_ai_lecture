# 교육 샘플 데이터와 산출물 계보

## 공개 한국 영수증

- 입력: `sample_docs/public_receipts/korea/taebaek_restaurant_2025_redacted.png`
- 유형: 공개 실제 발행 영수증의 비식별 파생본
- 출처·라이선스·해시: `sample_docs/public_receipts/metadata.json`
- 준비 OCR 텍스트: 사람이 원본과 대조한 교육용 fixture
- 주의: 현재 실행에서 모델을 호출한 결과가 아님

준비 결과에 남길 최소 필드:

```json
{
  "fixture_type": "human_verified_transcription_fixture",
  "input_file": "taebaek_restaurant_2025_redacted.png",
  "input_sha256": "19227c...",
  "engine": "not_executed",
  "engine_version": "not_applicable",
  "target_technology": "PaddleOCR Korean",
  "recorded_at": "2026-07-28",
  "reviewer": "course maintainer",
  "disclaimer": "현재 실행에서 모델을 호출한 결과가 아닙니다."
}
```

## 합성 영수증

- `sample_docs/receipt_sample.png`
- `sample_docs/receipt_low_quality.png`
- `sample_docs/receipt_sample.pdf`
- `sample_outputs/ocr_result.json`: 실행 모델이 없는 합성 bbox fixture
- `sample_outputs/paddleocr_vl_result.json`: PaddleOCR-VL 출력 형태를 설명하는 합성 fixture
- `sample_outputs/extracted_result.json`: 위 합성 영수증의 규칙 추출 정답

용도: 입력 품질 비교, 독립 단위 테스트, 공개 실제 영수증과 분리된 설명용 데이터<br>
표시: “교육용 합성 문서 · 실제 개인정보 없음”

## 확장 문서 3종

| 유형 | 실물형 사진 | 정답 JSON | 실제 형식 파일 |
| --- | --- | --- | --- |
| 견적서 | `extensions/quotation_photo.png` | `sample_outputs/extensions/quotation.json` | `formats/quotation.xlsx` |
| 신청서 | `extensions/application_form_photo.png` | `sample_outputs/extensions/application.json` | `formats/application_form.docx` |
| 거래명세서 | `extensions/transaction_statement_photo.png` | `sample_outputs/extensions/transaction_statement.json` | `formats/transaction_statement.pdf` |

추가 PPT: `sample_docs/formats/table_summary.pptx`

세 문서는 모두 2026-07-28에 코드로 생성한 교육용 합성 문서다. 실제 사람·회사·거래 데이터가 없다. 사진형 파일은 책상 위에서 한 장을 촬영한 상황을 재현하며, 일반 정확도 평가용 데이터셋이 아니다.

## 산출물 해석

- `synthetic_fixture`: 형식·검증 코드를 설명하기 위한 합성 데이터
- `human_verified_transcription_fixture`: 공개 원본을 사람이 대조해 옮긴 준비 결과이며 모델 실행 기록이 아님
- `recorded_model_run`: 특정 입력·모델·버전·시각이 기록된 실제 실행 결과
- `live_inference`: 현재 실행에서 모델이 처리한 결과

이 네 유형을 같은 `model result`로 부르지 않는다.
