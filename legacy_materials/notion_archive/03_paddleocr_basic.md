# 기존 교재 3. PaddleOCR 설치 및 기본 사용법

- 원본: [Paddle OCR 설치 및 기본 사용법](https://app.notion.com/p/282707c7ae75818c86d0d520f9034122)
- Notion 조회 시점의 최종 수정일: 2025-10-15
- 연결 실습: `notebooks/Lab03_PaddleOCR.ipynb`

## 학습 목표

- PaddleOCR 설치와 초기화
- 한글·영어 혼합 이미지 인식
- 텍스트, 신뢰도, 바운딩 박스 확인
- 인식 결과 시각화

## 실습 흐름

1. PaddlePaddle과 PaddleOCR 설치
2. Colab 한글 폰트 설정
3. 삼성전자 품의서 형식의 가상 샘플 이미지 생성
4. `PaddleOCR(...).predict()` 실행
5. `rec_texts`, `rec_scores`, `dt_polys` 확인
6. 원본 이미지에 바운딩 박스와 텍스트 표시
7. 신뢰도 80%를 기준으로 초록색과 주황색 구분

## 코드 자산

```python
ocr = PaddleOCR(use_textline_orientation=True, lang="korean")
result = ocr.predict("sample_doc.jpg")

ocr_result = result[0]
texts = ocr_result["rec_texts"]
scores = ocr_result["rec_scores"]
boxes = ocr_result["dt_polys"]
```

## 개편 판단

- 텍스트·신뢰도·좌표라는 OCR 결과 구조는 2교시에 재사용
- 특정 회사 품의서는 개인정보 없는 중립 샘플로 교체
- 80% 고정 기준은 교육 예시로만 사용하고 업무 위험도에 따라 달라짐을 설명
- 설치 명령과 PaddleOCR API는 Python 3.12.9 환경에서 다시 실행 검증
- 설치 실패 시 동일 구조의 mock OCR 결과를 제공
- 상세 시각화는 시간 여유가 있을 때의 확장 실습으로 이동

판정: **수정 후 2교시 핵심 실습으로 재사용**
