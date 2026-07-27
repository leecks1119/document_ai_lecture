# 2교시. OCR 결과를 눈으로 확인하기

> 준비된 OCR 결과를 원본 영수증과 비교하고 틀릴 가능성이 있는 값을 찾습니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/02_ocr_basic.ipynb)

## 1. 학습 목표

- OCR 결과의 텍스트·위치·신뢰도를 구분할 수 있다.
- 입력 품질이 OCR 결과에 영향을 줄 수 있음을 설명할 수 있다.
- 신뢰도를 정답으로 오해하지 않고 검토 신호로 사용할 수 있다.

## 2. 이번 교시의 결과물

- `ocr_text.txt`: OCR 결과에서 텍스트만 읽기 순서대로 저장한 파일

## 3. 시작하기 전에

### 선수 지식

- 리스트와 `for` 반복문의 의미를 알고 있으면 충분하다.

### 준비 파일

- [정상 영수증](../sample_docs/receipt_sample.png)
- [저품질 영수증](../sample_docs/receipt_low_quality.png)
- [준비된 OCR 결과](../sample_outputs/ocr_result.json)
- [2교시 Colab 노트북](../colab/02_ocr_basic.ipynb)

필수 실습은 준비된 OCR 결과를 사용한다. EasyOCR 모델 다운로드는 선택 실습이다.

## 4. 핵심 개념

### 4.1 OCR 결과에는 세 가지 정보가 있다

OCR 엔진은 글자만 반환할 수도 있지만, 많은 경우 다음 정보를 함께 준다.

- 위치: 문서의 어느 영역에서 읽었는가
- 텍스트: 무엇이라고 읽었는가
- 신뢰도: 엔진이 결과를 얼마나 안정적으로 냈는가를 나타내는 내부 점수

![바운딩 박스, 인식 텍스트, 신뢰도로 이루어진 OCR 결과](assets/02/01_ocr_anatomy.svg)

### 4.2 입력 품질이 결과에 영향을 준다

흐림, 기울기, 잘림, 작은 글자는 오류를 만들 수 있다. 전처리가 항상 좋아지는 것도 아니므로 원본과 결과를 직접 비교한다.

![정상 영수증과 기울고 흐린 영수증 비교](assets/02/02_quality_compare.svg)

### 4.3 신뢰도는 정답표가 아니다

> **쉬운 비유**
> OCR 결과는 학생이 칠판을 받아 적은 노트와 같다. 대부분 맞아도 숫자나 글자가 틀릴 수 있다.

비유의 한계: 신뢰도는 시험 점수나 정답 확률과 완전히 같지 않으며 OCR 엔진마다 의미가 다를 수 있다. 높은 점수여도 총액처럼 중요한 값은 원문과 비교한다.

## 5. 전체 실습 흐름

```text
준비된 OCR JSON 불러오기
  → 완성된 함수로 바운딩 박스 확인
  → 텍스트만 줄 단위로 합치기
  → 원본과 비교
  → ocr_text.txt 저장
```

## 6. 단계별 실습

### 실습 1. OCR 초안에서 텍스트 저장하기

`draw_boxes()`는 노트북에 완성 코드로 제공된다. 학습자는 다음 텍스트 결합 부분만 확인한다.

```python
ocr_text = "\n".join(item["text"] for item in ocr_result)
print(ocr_text)
```

```python
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

외부 파일을 읽지 못하면 노트북 안의 `MOCK_OCR_RESULT`를 사용한다. 출력에 `MOCK OCR 결과`라고 표시되는지 확인한다.

### 선택 실습. EasyOCR 실행

모델 다운로드가 가능할 때만 실행한다. 실패하면 필수 실습 결과에는 영향이 없다.

```python
RUN_OPTIONAL_EASYOCR = False
```

`True`로 바꾸면 EasyOCR를 설치하고 합성 영수증을 읽는다. 회사 문서나 개인정보 문서는 사용하지 않는다.

## 7. Codex 활용

### 요청 목표

신뢰도를 자동 승인 기준으로 오해하는 코드가 없는지 확인한다.

### 실습 프롬프트

```text
목표: OCR 결과 확인 코드를 초보자 관점에서 검토해줘.
맥락: 결과에는 box, text, confidence가 있어.
제약조건: confidence만 보고 값을 자동 승인하는 코드는 제안하지 마.
완료 기준: 원본과 다시 비교해야 할 위치를 두 문장으로 설명해줘.
```

### 생성 결과 확인

- 신뢰도를 정답 확률이라고 단정하지 않았는가?
- 금액과 날짜의 원문 확인을 제안했는가?

## 8. 문제 해결

| 증상 | 원인 | 해결 방법 |
| --- | --- | --- |
| OCR 결과 파일을 찾을 수 없음 | Colab에는 저장소 전체가 없음 | 노트북 내장 `MOCK_OCR_RESULT` 사용 |
| EasyOCR 설치가 오래 걸림 | 모델·패키지 다운로드 | 선택 셀을 중지하고 준비된 결과 사용 |
| 한글이 상자로 보임 | 표시용 폰트 문제 | 텍스트 출력으로 값을 확인하고 실습 계속 |

## 9. 형성평가

1. 신뢰도가 높으면 값을 자동으로 정답 처리해도 되는가?
2. OCR 오류를 만들 수 있는 입력 상태 두 가지는 무엇인가?

<details>
<summary>정답 보기</summary>

1. 아니다. 신뢰도는 검토 순서를 정하는 참고 신호다.
2. 흐림, 기울기, 잘림, 작은 글자 중 두 가지.

</details>

## 10. 핵심 요약

- OCR 결과는 텍스트·위치·신뢰도로 볼 수 있다.
- 저품질 입력은 오류 가능성을 높인다.
- 신뢰도만으로 중요한 값을 확정하지 않는다.

## 11. 완료 체크리스트

- [ ] OCR 결과의 세 요소를 구분했다.
- [ ] 원본과 OCR 텍스트를 비교했다.
- [ ] `ocr_text.txt`를 만들었다.

## 12. 다음 교시 예고

3교시에서는 OCR 초안을 원문을 잃지 않으면서 키-값과 반복 품목으로 정리한다.

## 참고 자료

- [EasyOCR Tutorial](https://www.jaided.ai/easyocr/tutorial/)
- [EasyOCR 1.7.2 Release](https://github.com/JaidedAI/EasyOCR/releases/tag/v1.7.2)
- [Google Document AI 지원 파일](https://docs.cloud.google.com/document-ai/docs/file-types)
