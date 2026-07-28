# 공개 영수증 자료 검토표

기준일: 2026-07-27

## 채택 원칙

교육 자료에 이미지를 포함하려면 실제 발행 여부, 공식 출처, 이미지 이용조건, 식별정보 가림 상태를 모두 확인한다. “무료 다운로드”나 공개 저장소라는 사실만으로 복사하지 않는다.

## 한국 영수증

| 자료 | 확인 내용 | 과정 사용 |
| --- | --- | --- |
| [태백 음식점 영수증](https://commons.wikimedia.org/wiki/File:Receipt_taebaek_restaurant_IMG_2614_modified.jpg) | 2025-10-04 실제 영수증. Commons 설명상 Public Domain(`PD-ineligible`). 원본에 일부 모자이크가 있으나 전화번호가 남아 있음 | **1교시 채택:** 전화번호·거래 식별 영역 추가 가림, 하단 결제 영역 제외, 메타데이터 제거 파생본 |
| [2013년 서울 마트 영수증](https://commons.wikimedia.org/wiki/File:SK_Korea_tour_%E9%A6%96%E7%88%BE_supermarket_official_receipt_computer_print_out_in_Korean_language_only_July-2013.JPG) | 실제 한국 영수증, Woodarelaisku, CC BY-SA 3.0. 사업자 정보와 촬영 GPS 포함 | **링크만 제공:** 저장소 복사·자동 다운로드 제외 |
| [Korean Receipts Dataset](https://huggingface.co/datasets/HumynLabs/Korean_Receipts_Dataset) | 한국어 20장, CC BY 4.0. 데이터 카드가 현장 수집·모의·크라우드소싱 혼합이라고 밝히나 파일별 유형 구분 없음 | **링크만 제공:** 개별 이미지를 실물이라고 단정하지 않음 |
| [KORIE 논문](https://www.mdpi.com/2227-7390/14/1/187) | 748장 한국 소매 영수증 벤치마크를 소개 | **연구 참고:** 공개 파일 위치·이미지 재배포 조건을 확인하기 전 저장소 포함 안 함 |

## 해외·보조 자료

| 데이터셋 | 자료와 언어 | 권리 판단 | 과정 사용 |
| --- | --- | --- | --- |
| [CORD v2](https://github.com/clovaai/cord) | 인도네시아 실물 영수증 공개본 1,000장, OCR·레이아웃·구조 주석 | 공식 저장소가 CC BY 4.0으로 배포 | **해외 비교:** test 0~2 세 장만 출처와 함께 포함 |
| [SROIE](https://rrc.cvc.uab.es/?ch=13) | 말레이시아 영문 영수증 1,000장 | 공식 대회 자료. 공식 원본 확보·표시 확인 필요 | **링크만 제공** |
| [WildReceipt](https://www.paddleocr.ai/main/en/datasets/kie_datasets.html) | 영수증 1,267장 학습·472장 평가, 26개 KIE 범주 | 공식 문서에 공개 데이터로 소개되나 원 이미지 라이선스 명시가 충분하지 않음 | **재배포 제외** |
| [Humans in the Loop Receipt OCR](https://humansintheloop.org/resources/datasets/free-receipt-ocr-dataset/) | ExpressExpense 샘플 192장, CC0 1.0 | 실제 구매 영수증이 아닌 통제된 샘플 | **실물 실습에서 제외** |

## 1교시 대표 자료

`sample_docs/public_receipts/korea/taebaek_restaurant_2025_redacted.png` 한 장을 세 기술에 동일하게 사용한다.

```text
OCR
  픽셀 → 텍스트 영역 탐지 → 문자 인식 → 텍스트·좌표·신뢰도

VLM
  이미지+지시 → 시각·배치와 언어 관계 해석 → 표·Markdown·초안 JSON

Document AI
  입력 품질 → OCR/VLM/혼합 → 업무 스키마 → 검증 → 사람 확인 → 저장
```

준비된 결과는 모두 `교육용 예시 — 실제 모델 실행 결과가 아님`으로 표시한다.

## 개인정보와 재현성

- 파생본과 원본의 SHA-256, 가림·자르기·메타데이터 제거 내용을 `sample_docs/public_receipts/metadata.json`에 기록한다.
- 가려진 영역을 복원하거나 구매자를 추론하지 않는다.
- 공개 영수증을 실제 회사 문서의 외부 업로드 허가로 해석하지 않는다.
- `python tools/download_public_receipts.py`로 공식 원본 해시를 확인하고 같은 파생본을 재현한다.
