# 2교시. OCR 기반 텍스트 추출 실습

> 각자 식별정보를 가린 영수증 한 장에 실제 OCR을 시도하고, 읽은 글자·위치·신뢰도를 원본과 비교합니다.
>
> **핵심 메시지:** OCR 결과는 정답이 아니라 컴퓨터가 읽어 본 초안이므로 숫자와 읽기 순서를 반드시 원본과 대조해야 합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/02_ocr_basic.ipynb)

## 1. 학습 목표

- 한 번에 영수증 한 장을 PP-OCRv5 Korean으로 판독할 수 있다.
- OCR 결과에서 텍스트·위치·신뢰도를 확인할 수 있다.
- 날짜·금액의 오인식과 읽기 순서 오류를 원본에서 찾아 표시할 수 있다.

## 2. 이번 교시의 결과물

- `ocr_result.json`: OCR 원문·위치·신뢰도와 원본 대조 표시를 담은 파일

## 3. 시작하기 전에

### 선수 지식

- 리스트와 `for` 반복문의 의미를 알면 충분하다.

### 준비 파일

- 식별정보를 가린 점심·커피 영수증 또는 사진첩의 영수증 한 장
- [공개 한국 실물 영수증 대체 샘플](../sample_docs/public_receipts/korea/taebaek_restaurant_2025_redacted.png)
- [저품질 영수증](../sample_docs/receipt_low_quality.png)
- [준비된 OCR 결과](../sample_outputs/ocr_result.json)
- [2교시 Colab 노트북](../colab/02_ocr_basic.ipynb)

카드번호·승인번호·현금영수증 번호·전화번호·회원번호는 촬영 전이나 업로드 전에 가린다. 실제 OCR을 먼저 시도하되 3분 안에 설치·모델 다운로드·실행이 끝나지 않으면 준비된 `ocr_result.json`으로 전환한다. 전환해도 원본 대조 실습은 동일하게 진행한다.

## 4. 핵심 개념

### 4.1 OCR 결과는 글자만이 아니다

PaddleOCR 결과에서 이번 시간에 볼 값은 세 가지다.

- `rec_texts`: 읽은 문자열
- `rec_polys`: 문자열이 있던 위치
- `rec_scores`: 모델의 인식 신뢰도

![위치, 텍스트, 신뢰도로 이루어진 PaddleOCR 결과](assets/02/01_ocr_anatomy.svg)

### 4.2 한국어는 맞는 모델 설정을 고른다

이 과정은 `PaddleOCR 3.7`의 한국어 설정인 `lang="korean"`과 `PP-OCRv5`를 사용한다. 최신 숫자가 붙었다고 모든 언어를 지원하는 것은 아니다. 2026-07-27 현재 PP-OCRv6의 공식 언어 목록에는 한국어가 없으므로 이 샘플에는 PP-OCRv5를 쓴다.

![정상 영수증과 기울고 흐린 영수증 비교](assets/02/02_quality_compare.svg)

### 4.3 신뢰도는 정답표가 아니다

> **쉬운 비유**
> OCR은 칠판을 받아 적는 학생이다. 글씨를 자신 있게 적어도 숫자를 틀릴 수 있다.

비유의 한계: 신뢰도는 시험 점수나 정답 확률과 같지 않다. 총액·날짜는 점수가 높아도 원본과 비교한다.

## 5. 전체 실습 흐름

```text
식별정보를 가린 영수증 한 장 선택
  → PP-OCRv5 Korean 실행 또는 3분 뒤 준비 결과로 전환
  → 위치·텍스트·신뢰도 확인
  → 원본과 비교
  → 잘못 읽은 값과 순서 표시
  → ocr_result.json 저장
```

## 6. 단계별 실습

### 실습 1. OCR 결과를 원본과 대조해 저장하기

실제 OCR 또는 준비 결과의 각 항목에 원본 대조 표시를 추가한다. `matches_source`는 원본을 직접 확인한 뒤 `True` 또는 `False`로 바꾼다.

```python
import json

reviewed = [
    {**item, "matches_source": None, "review_note": ""}
    for item in OCR_RESULT
]
with open("ocr_result.json", "w", encoding="utf-8") as file:
    json.dump(reviewed, file, ensure_ascii=False, indent=2)
```

**기대 결과**

- 각 텍스트에 위치·신뢰도·원본 대조 표시가 함께 남는다.
- Colab 파일 영역에 `ocr_result.json`이 생긴다.

**3분 복구 경로**

실제 OCR이 3분 안에 실행되지 않으면 `MOCK_OCR_RESULT`로 같은 확인·대조·저장 과정을 수행한다. 출력에는 `준비된 OCR 결과`라고 표시한다.

**기본: 실제 PaddleOCR 실행**

```python
RUN_PADDLEOCR = True
```

PaddleOCR 3.7과 PP-OCRv5 Korean으로 식별정보를 가린 영수증 한 장을 읽는다. 개인 영수증을 Colab에 올리기 어렵다면 공개 한국 실물 영수증을 사용한다.

## 7. 실습 결과 확인

- 텍스트·위치·신뢰도를 원본의 같은 영역과 비교했는가?
- 날짜와 합계가 틀린 경우 `matches_source=False`로 표시했는가?
- 준비 결과를 사용했다면 실제 OCR 결과처럼 표시하지 않았는가?
- `ocr_result.json`을 Colab에서 내려받았는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 결과 파일이 없음 | 외부 파일 미업로드 | 공개 샘플 또는 내장 `MOCK_OCR_RESULT` 사용 |
| 설치가 3분 넘게 걸림 | 패키지·모델 다운로드 | 실행을 중지하고 준비 결과로 전환 |
| 한글 결과가 좋지 않음 | 언어 설정 불일치 | `lang="korean"`, `PP-OCRv5` 확인 |

## 9. 형성평가

1. 한국어 영수증에는 어떤 OCR 버전을 사용하는가?
2. 높은 신뢰도의 총액을 자동 확정해도 되는가?

<details>
<summary>정답 보기</summary>

1. PaddleOCR 3.7에서 `lang="korean"`, `PP-OCRv5`를 사용한다.
2. 아니다. 중요한 값은 원본과 비교한다.

</details>

## 10. 핵심 요약

- PaddleOCR 결과에서 텍스트·위치·신뢰도를 본다.
- 한국어 샘플에는 PP-OCRv5 설정을 사용한다.
- 신뢰도는 검토 신호이지 정답표가 아니다.

## 11. 완료 체크리스트

- [ ] 세 결과 요소를 구분했다.
- [ ] 원본과 OCR 결과를 비교했다.
- [ ] 원본 대조 표시가 포함된 `ocr_result.json`을 만들었다.

## 12. 다음 교시 예고

3교시에서는 OCR 초안을 원문을 잃지 않으면서 문서 구조별로 정리한다.

## 참고 자료

- [PaddleOCR 3.x OCR 파이프라인](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [PP-OCRv5 다국어 인식](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
