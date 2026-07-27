# 공개 실물 영수증 실습 자료

이 폴더에는 출처와 이용조건을 확인한 **실제 발행 영수증**만 둔다. 1교시 대표 입력은 한국 영수증이며, 해외 자료는 비교용으로만 사용한다.

## 한국 대표 자료

| 파일 | 원본 | 개인정보 처리 | 수업 용도 |
| --- | --- | --- | --- |
| `korea/taebaek_restaurant_2025_redacted.png` | 2025-10-04 태백 음식점 영수증 | 원본의 모자이크 유지, 전화번호·거래 식별 영역 추가 가림, 하단 결제 영역 제외, 메타데이터 제거 | 1교시 OCR·VLM·Document AI 비교 |

원본은 Wikimedia Commons의 [Receipt taebaek restaurant IMG 2614 modified.jpg](https://commons.wikimedia.org/wiki/File:Receipt_taebaek_restaurant_IMG_2614_modified.jpg)이다. Commons 파일 설명은 이 자료를 Public Domain(`PD-ineligible`)으로 표시한다.

> 원본 Source/Author: 이태리집(태백시 소재), uploader: Choikwangmo25. 교육용 파생본의 이용은 원 출처의 보증이나 후원을 의미하지 않는다.

이 이미지는 합성 문서가 아니라 연락처·거래 식별정보를 가린 공개 실물 영수증 파생본이다. 공개 사업자명·발행 시각·품목·금액은 남아 있으며, 모델 성능의 대표값을 계산할 수 있는 정답 데이터셋은 아니다.

## 해외 비교 자료

`cord_v2/`에는 NAVER Clova의 CORD v2 공개본 중 인도네시아 영수증 세 장이 들어 있다. 1교시 대표 예제로 사용하지 않고, 언어·배치가 다른 문서의 선택 비교에만 사용한다.

- 데이터셋: [CORD](https://github.com/clovaai/cord)
- 공식 배포: [NAVER Clova CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- 저자: Seunghyun Park 외
- 라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 변경: 이미지 내용 변경 없음, 파일명만 수업용으로 변경

## 링크로만 제공하는 한국 사례

[2013년 서울 마트 영수증](https://commons.wikimedia.org/wiki/File:SK_Korea_tour_%E9%A6%96%E7%88%BE_supermarket_official_receipt_computer_print_out_in_Korean_language_only_July-2013.JPG)은 실제 한국 영수증이고 CC BY-SA 3.0이다. 그러나 대표자명·사업자번호·주소·전화번호와 촬영 GPS가 포함되어 있어 저장소에는 복사하지 않는다.

## 안전 원칙

- 가려진 영역을 복원하거나 개인·결제 정보를 추론하지 않는다.
- 공개 영수증이라고 해서 실제 회사 문서를 Colab이나 공개 앱에 올려도 된다는 뜻은 아니다.
- 준비된 OCR·VLM 예시는 실제 모델 실행 결과처럼 표시하지 않는다.
- 새 자료는 [공개 영수증 검토표](../../../docs/public_receipt_datasets.md)에 출처·권리·수정·해시를 먼저 기록한다.

## 재현

다음 명령은 한국 파생본을 공식 원본에서 다시 만들고 CORD 세 장을 공식 배포본에서 검증한다.

```bash
python tools/download_public_receipts.py
```
