# 2교시. OCR 기반 텍스트 추출 실습

> **이번 교시의 한 문장:** OCR 결과는 정답이 아니라 원본과 대조할 판독 결과입니다.

[Colab 실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/02_ocr_basic.ipynb)

## 60분 뒤 남길 것

- 공개 한국 영수증에 실제 OCR을 실행합니다.
- 텍스트·좌표·신뢰도를 원본 위에서 확인합니다.
- `course_outputs/ocr_result.json`과 `ocr_boxes.png`를 만듭니다.

## 안전 확인

Google Colab도 외부 클라우드입니다. 오늘은 제공된 비식별 공개 샘플만 사용합니다. 개인 영수증은 조직 승인과 완전한 비식별화를 거친 선택 실습에서만 사용합니다.

![개인정보와 거래 식별 영역을 가린 공개 한국 영수증](../sample_docs/public_receipts/korea/taebaek_restaurant_2025_redacted.png)

## 개념 10%: OCR 결과의 세 요소

- `text`: 모델이 읽은 문자열
- `box`: 문자열이 있던 위치
- `confidence`: 모델 내부의 확신 신호

신뢰도는 정답 확률이나 업무 승인 점수가 아닙니다. 금액·날짜처럼 영향이 큰 값은 원본 대조가 필요합니다.

![위치, 텍스트, 신뢰도로 이루어진 PaddleOCR 결과](assets/02/01_ocr_anatomy.svg)

### PP-OCRv5 Korean을 쓰는 이유

PaddleOCR 3.7의 일반 OCR 기본 계열은 PP-OCRv6이지만, 한국어 전용 인식 모델은 공식 문서의 `korean_PP-OCRv5_mobile_rec` 계열을 사용합니다. 이 과정은 `lang="korean"`, `ocr_version="PP-OCRv5"`를 명시합니다.

### 같은 문서도 품질에 따라 달라집니다

![선명한 합성 영수증](../sample_docs/receipt_sample.png)

![기울고 흐린 같은 합성 영수증](../sample_docs/receipt_low_quality.png)

![정상 영수증과 기울고 흐린 영수증 비교](assets/02/02_quality_compare.svg)

## 실습 90%

### 1. 실행 모드를 확인합니다

노트북의 기본값은 `LIVE`입니다.

```text
요청 모드: LIVE
```

교육용 오프라인 검증이나 네트워크 장애 때만 `PREPARED_FALLBACK`을 사용합니다.

준비 결과는 현재 모델 실행이 아니므로 `confidence`가 `null`입니다. 실제 실행에서 얻은 것처럼 임의 신뢰도를 붙이지 않습니다.

### 2. 실제 PaddleOCR을 실행합니다

노트북이 PaddleOCR 3.7과 한국어 모델을 설치한 뒤 공개 영수증을 처리합니다. 첫 모델 다운로드는 시간이 걸릴 수 있습니다.

정상 화면:

```text
실행 모드: LIVE
판독 영역: ...
```

3분을 넘기거나 설치·모델 저장소 오류가 나면 실행을 중지합니다. 복구 경로는 전환 사유를 숨기지 않습니다.

```text
실행 모드: PREPARED_FALLBACK
복구 사유: ...
```

### 3. 바운딩 박스를 원본과 비교합니다

`ocr_boxes.png`에서 다음 세 곳을 찾습니다.

- 상호명
- 거래 일시
- 합계 금액

텍스트가 맞아도 박스가 다른 줄을 가리키면 근거 연결에 실패한 것입니다.

### 4. 원본 대조 상태를 저장합니다

`ocr_result.json`에는 실행 모드와 각 영역의 검토 칸이 남습니다.

```json
{
  "source_mode": "LIVE",
  "items": [
    {
      "text": "합계 금액 76,000",
      "confidence": 0.96,
      "matches_source": null,
      "review_note": ""
    }
  ]
}
```

마지막에 다음 문구가 보여야 합니다.

```text
CHECKPOINT 1/1 PASS: ... course_outputs/ocr_result.json course_outputs/ocr_boxes.png
```

두 파일은 자동으로 다운로드됩니다. 다음 교시 새 Colab 창에서 `ocr_result.json`을 선택해야 자신의 결과가 이어집니다.

## 통과 기준

- `source_mode`가 `LIVE` 또는 `PREPARED_FALLBACK`으로 명확히 표시됩니다.
- `ocr_result.json`과 `ocr_boxes.png`가 생성되고 로컬에 다운로드되었습니다.
- 높은 신뢰도도 원본 확인 전에는 승인할 수 없다고 설명할 수 있습니다.
- 실제 실행과 복구 결과를 구분할 수 있습니다.

## 막혔을 때

- 설치·다운로드가 3분을 넘으면 중지하고 준비 결과 경로로 계속합니다.
- OCR 영역이 0개면 입력 이미지가 열렸는지 먼저 확인합니다.
- `LIVE_ERROR`나 설치 오류 문구를 지우지 말고 보존합니다.
- 개인 문서로 다시 시도하지 않습니다.

다음 교시에는 이 결과를 불러와 원문을 보존한 채 키-값과 반복 품목으로 정리합니다.

## 참고 자료

공식 근거는 [과정 참고자료와 적용 범위](../docs/course_references.md)의 2교시 표에서 확인할 수 있습니다.
