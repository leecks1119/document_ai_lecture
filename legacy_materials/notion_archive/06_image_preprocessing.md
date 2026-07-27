# 기존 교재 6. 이미지 전처리

- 원본: [노이즈 제거, 이진화, 기울기 보정](https://app.notion.com/p/282707c7ae75816fa0d1e2c95fafc59f)
- Notion 조회 시점의 최종 수정일: 2025-10-16

## 학습 목표

- 전처리의 필요성 이해
- 노이즈 제거, 이진화, 기울기 보정 실습
- 전처리 전후 OCR 결과 비교

## 포함된 개념

### 노이즈 제거

- Median Filter
- Gaussian Blur
- Non-local Means Denoising

### 이진화

- Otsu
- Adaptive Threshold
- Sauvola
- 균일 조명, 불균일 조명, 희미한 텍스트에 따른 선택 흐름

### 기울기 보정

- Canny와 Hough Line Transform
- 회전 변환 행렬
- Projection Profile을 이용한 각도 탐색

### 통합 파이프라인

- 품질 평가
- 기울기 보정
- 노이즈 제거
- 그레이스케일
- CLAHE 대비 향상
- Otsu와 Adaptive Threshold 비교
- 모폴로지 연산
- OCR 전후 A/B 테스트

## 코드 자산

- `detect_skew_angle`
- `correct_skew`
- `sauvola_threshold`
- `DocumentPreprocessor`
- 전처리 전후 OCR 정확도 비교 함수
- 품질 지표로 blur, skew, noise를 계산하는 아이디어

## 재검증이 필요한 내용

- 전처리 전 65~75%, 후 85~95% 정확도
- 월 10,000건 기준 ROI와 1.1개월 회수 기간
- 기대 정확도 15~30%p 향상
- 임계값과 파라미터가 모든 문서에 적용된다는 인상
- 일부 PaddleOCR 호출이 구버전 API 형식

## 개편 판단

- 전체 60분 실습은 새 8시간 과정의 핵심 범위를 벗어남
- 기울기·노이즈·조명이 OCR에 미치는 영향은 2~3교시 설명에 재사용
- 저품질 문서 한 장에 “원본 그대로”와 “간단 전처리”만 비교
- 고급 전처리 클래스와 ROI 계산은 강사용 참고 또는 심화 과정으로 이동

판정: **핵심 원리와 전후 비교만 재사용, 고급 구현은 참고 자료**
