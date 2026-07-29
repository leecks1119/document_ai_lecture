# 8교시. 실무 적용 시나리오 설계 및 최종 정리

마지막 시간에는 영수증이 아닌 견적서·신청서·거래명세서 사진을 실제 문서 VLM에
넣어 봅니다. 영수증에서 배운 원리가 다른 업무 문서에도 이어지는지 확인합니다.

> **이번 시간의 도착점:** 업무 문서 사진 한 장을
> `PaddleOCR-VL-1.6-0.9B`로 직접 읽고, 작은 PoC 후보 카드를 완성합니다.

[Colab 실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/master/colab/08_business_application.ipynb)

## 먼저 문서 한 장을 고릅니다

수업에는 개인정보가 없는 실물형 합성 사진 세 장이 준비되어 있습니다.

- 견적서 사진
- 신청서 사진
- 거래명세서 사진

![실물형 견적서 사진](../sample_docs/extensions/quotation_photo.png)

처음에는 거래명세서를 선택하고, 시간이 남으면 다른 문서로 다시 실행합니다.
비식별 개인 이미지나 공개 이미지 URL도 한 장씩 사용할 수 있습니다.

## 직접 해보기

### 1. 실제 VLM을 실행합니다

Colab 런타임을 `T4 GPU`로 설정한 뒤 다음 셀을 실행합니다.

```python
business_vlm_result = parse_with_paddleocr_vl(
    INPUT_PATH,
    engine="transformers",
    device="gpu",
)
```

이 함수는 전체 `PaddleOCR-VL-1.6` 파이프라인을 실행하며, 내부 VLM은
`PaddleOCR-VL-1.6-0.9B`입니다.

실제 Colab 실행에서는 다음 문구를 확인합니다.

```text
실제 모델 실행: True
```

이어지는 Markdown에서 제목, 표, 금액, 날짜가 어떤 순서로 복원됐는지 봅니다.
잘못 읽힌 값도 지우지 말고 원본과 비교합니다.

### 2. Office 파일 형식도 직접 열어 봅니다

노트북은 다음 네 파일을 `office_format_samples.zip`으로 제공합니다.

- Excel 견적서
- Word 신청서
- PDF 거래명세서
- PowerPoint 표 캡처

각 파일을 직접 열고 다음 차이를 확인합니다.

| 형식 | 먼저 확인할 구조 |
| --- | --- |
| Excel | 셀, 수식, 병합 셀 |
| Word | 문단, 표, 본문 안 이미지 |
| PDF | 선택 가능한 텍스트와 스캔 이미지 |
| PowerPoint | 도형 순서와 표 캡처 |

원본 구조가 남아 있다면 OCR보다 원본 구조를 먼저 읽는 편이 유리합니다. 이미지로
평면화된 문서는 OCR이나 VLM이 필요합니다.

### 3. 작은 PoC 카드 만들기

복잡한 사업계획서를 작성하지 않습니다. 다음 네 가지만 정합니다.

1. 자동화할 문서 한 종류
2. 추출할 핵심 필드
3. 원본을 확인할 사람
4. 자동 저장을 중단할 조건

예시는 다음과 같습니다.

```text
대상: 거래명세서
추출: 거래처, 거래일, 품목, 공급가액, 세액, 합계
검토자: 정산 담당자
중단: 필수값이나 원문 근거가 없으면 Excel 저장 금지
```

셀을 실행하면 실제 VLM 결과 일부가 포함된 다음 파일이 만들어집니다.

```text
course_outputs/poc_candidate_card.md
```

![사람 검토를 포함한 업무 확장 흐름](assets/08/01_human_review.svg)

## 이번 시간의 정리

실제 모델 결과를 한 번 확인한 뒤 문서 종류·핵심 필드·검토자·저장 중단 조건을
정하면 작은 PoC의 범위를 구체적으로 설명할 수 있습니다.

## 완료 확인

- 업무 문서 사진 한 장을 선택했습니다.
- 실제 PaddleOCR-VL 결과와 Markdown을 확인했습니다.
- Excel·Word·PDF·PPT 샘플을 직접 열어 봤습니다.
- 검토자와 자동 저장 중단 조건을 정했습니다.
- `office_format_samples.zip`을 내려받아 네 형식을 열어 봤습니다.
- `course_outputs/poc_candidate_card.md`를 만들었습니다.

강의를 마친 뒤에는 다음 질문에 답할 수 있어야 합니다.

> “영수증으로 해보니 원리를 알겠다. 우리 회사의 견적서나 신청서,
> 거래명세서도 이런 식으로 자동화해볼 수 있겠는데?”

## 참고자료

- [PaddleOCR-VL 공식 사용법](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleX PaddleOCR-VL 파이프라인 설명](https://paddlepaddle.github.io/PaddleX/3.7/en/pipeline_usage/tutorials/ocr_pipelines/PaddleOCR-VL.html)
