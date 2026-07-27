# 2교시. PaddleOCR 결과를 눈으로 확인하기

> 한국어 영수증의 글자·위치·신뢰도를 원본과 비교합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/02_ocr_basic.ipynb)

## 1. 학습 목표

- PaddleOCR 결과에서 텍스트·위치·신뢰도를 찾을 수 있다.
- 입력 품질이 OCR 결과에 미치는 영향을 설명할 수 있다.
- 중요한 값은 신뢰도만 믿지 않고 원본과 비교할 수 있다.

## 2. 이번 교시의 결과물

- `ocr_text.txt`: OCR 텍스트를 읽기 순서대로 저장한 파일

## 3. 시작하기 전에

### 선수 지식

- 리스트와 `for` 반복문의 의미를 알면 충분하다.

### 준비 파일

- [정상 영수증](../sample_docs/receipt_sample.png)
- [저품질 영수증](../sample_docs/receipt_low_quality.png)
- [준비된 OCR 결과](../sample_outputs/ocr_result.json)
- [2교시 Colab 노트북](../colab/02_ocr_basic.ipynb)

기본 실습은 준비된 결과로 진행한다. 실제 모델 실행은 다운로드가 가능한 학습자만 선택한다.

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
합성 영수증과 준비된 OCR 결과 열기
  → 위치·텍스트·신뢰도 확인
  → 원본과 비교
  → 텍스트만 합치기
  → ocr_text.txt 저장
```

## 6. 단계별 실습

### 실습 1. OCR 텍스트 저장하기

노트북의 준비된 결과를 사용한다.

```python
ocr_text = "\n".join(item["text"] for item in MOCK_OCR_RESULT)
with open("ocr_text.txt", "w", encoding="utf-8") as file:
    file.write(ocr_text)
```

**기대 결과**

```text
샘플문구점
거래일자: 2026-07-27
연필 2개 × 1,000원 = 2,000원
노트 1개 × 3,000원 = 3,000원
합계: 5,000원
```

**mock 대체 경로**

외부 파일이나 모델을 읽지 못해도 `MOCK_OCR_RESULT`로 같은 확인·저장 과정을 수행한다. 출력에 `MOCK OCR 결과`가 표시된다.

**선택: 실제 PaddleOCR 실행**

```python
RUN_PADDLEOCR = False
```

`True`로 바꾸면 노트북이 PaddleOCR 3.7을 설치하고 합성 영수증을 읽는다. 회사 문서나 개인정보 문서는 사용하지 않는다.

## 7. Codex 활용

### 요청 목표

PaddleOCR 결과를 원본과 비교할 지점을 찾는다.

### 실습 프롬프트

```text
목표: PaddleOCR 결과 확인 코드를 초보자 관점에서 검토해줘.
맥락: 결과에는 rec_texts, rec_polys, rec_scores가 있어.
제약조건: 점수만 보고 값을 자동 승인하지 마.
완료 기준: 원본과 다시 비교할 값을 두 문장으로 알려줘.
```

### 생성 결과 확인

- 신뢰도를 정답 확률이라고 단정하지 않았는가?
- 금액과 날짜를 원본과 비교하라고 했는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| 결과 파일이 없음 | 외부 파일 미업로드 | 내장 `MOCK_OCR_RESULT` 사용 |
| 설치가 오래 걸림 | 패키지·모델 다운로드 | 선택 셀을 중지하고 기본 실습 계속 |
| 한글 결과가 좋지 않음 | 언어 설정 불일치 | `lang="korean"`, `PP-OCRv5` 확인 |

## 9. 형성평가

1. 한국어 합성 영수증에는 어떤 OCR 버전을 사용하는가?
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
- [ ] `ocr_text.txt`를 만들었다.

## 12. 다음 교시 예고

3교시에서는 OCR 초안을 원문을 잃지 않으면서 문서 구조별로 정리한다.

## 참고 자료

- [PaddleOCR 3.x OCR 파이프라인](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [PP-OCRv5 다국어 인식](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
