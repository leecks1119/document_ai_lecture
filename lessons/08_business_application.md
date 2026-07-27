# 8교시. 실무 적용 시나리오 설계 및 최종 정리

> **이번 교시의 한 문장:** 문서 종류가 바뀌어도 흐름은 같고, 추출 필드·검증 규칙·오류 영향이 달라집니다.

[Colab 실습 열기](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/08_business_application.ipynb)

## 60분 뒤 남길 것

- 견적서·신청서·거래명세서 실물형 사진을 비교합니다.
- Excel·Word·PDF·PPT 형식의 차이를 체험합니다.
- 첫 PoC 후보 한 가지의 필드·규칙·위험·검토자를 정합니다.
- `course_outputs/poc_candidate_card.md`를 만듭니다.

## 실물형 비식별 샘플 3종

모두 교육용 합성 문서이며 실제 개인정보가 없습니다.

### 견적서

![책상 위에서 촬영한 형태의 교육용 합성 견적서](../sample_docs/extensions/quotation_photo.png)

필드: 문서번호, 공급자, 수신, 견적일, 품목, 총액
핵심 규칙: 수량×단가, 공급가액+부가세
오류 영향: 구매 판단과 예산

### 신청서

![책상 위에서 촬영한 형태의 교육용 합성 신청서](../sample_docs/extensions/application_form_photo.png)

필드: 신청번호, 신청자, 소속, 신청 과정, 관리자 승인
핵심 규칙: 필수 동의와 승인 상태
오류 영향: 개인정보와 승인 누락

### 거래명세서

![책상 위에서 촬영한 형태의 교육용 합성 거래명세서](../sample_docs/extensions/transaction_statement_photo.png)

필드: 문서번호, 공급자, 거래일, 품목, 세액, 총액
핵심 규칙: 품목 합계·공급가액·세액·총액
오류 영향: 정산

## 개념 10%: 형식마다 잃기 쉬운 정보

| 형식 | 직접 읽을 수 있는 구조 | 확인할 예외 |
| --- | --- | --- |
| Excel | 셀·수식·시트 | 병합 셀·숨김 시트·숫자 서식 |
| Word | 문단·표·그림 | 머리글·텍스트박스·변경 추적·이미지 본문 |
| PDF | 텍스트층·페이지 | 스캔 혼합·암호·깨진 문자맵 |
| PPT | 도형·표·이미지 | 그룹 도형·읽기 순서·발표자 노트 |
| 표 캡처 | 픽셀 | 행·열·병합 셀 관계를 다시 복원 |

직접 열어 볼 파일:

- [견적서 Excel](../sample_docs/formats/quotation.xlsx)
- [이미지 기반 신청서 Word](../sample_docs/formats/application_form.docx)
- [텍스트층이 있는 거래명세서 PDF](../sample_docs/formats/transaction_statement.pdf)
- [표 캡처 설명 PowerPoint](../sample_docs/formats/table_summary.pptx)

같은 “문서”라도 파일 내부 구조가 있으면 먼저 직접 읽고, 이미지뿐일 때 OCR·VLM을 선택합니다.

## 실습 90%

### 1. 세 문서 사진을 나란히 비교합니다

Colab은 세 이미지를 표시합니다. 각 문서에서 반복 행, 키-값, 승인 칸, 총액을 찾습니다.

### 2. 첫 PoC 후보를 선택합니다

다음 다섯 항목을 1~5점으로 봅니다.

- 반복량
- 필드 안정성
- 오류 영향
- 예외 빈도
- 사람 검토 가능성

높은 위험을 “좋은 자동화 후보”로 착각하지 않습니다. 첫 PoC는 한 종류·한 장·승인된 비식별 입력으로 작게 시작합니다.

![추출과 검증 뒤 사람 확인을 거쳐 Excel이 확정되는 흐름](assets/08/01_human_review.svg)

### 3. PoC 카드 정답 예시를 실행합니다

```text
입력 제한: 승인된 비식별 한 장
최종 산출물: 사람 승인 후 Excel
제안: GO_SMALL 또는 REVIEW
```

마지막에 다음 문구가 보여야 합니다.

```text
CHECKPOINT 1/1 PASS: course_outputs/poc_candidate_card.md
```

![입력, 필드, 위험, 검토자, 저장, 삭제 항목이 있는 업무 적용 카드](assets/08/02_business_card.svg)

## PoC 통과 기준 예시

- 같은 양식 30장을 정답표와 비교합니다.
- 필드별 정확도뿐 아니라 수정률과 처리시간을 기록합니다.
- 오류는 자동 저장하지 않고 검토 대기열로 보냅니다.
- 개인정보·보존·삭제 정책을 먼저 승인받습니다.
- 성능이 낮을 때 모델만 바꾸지 않고 입력 품질·필드 정의·검증 규칙을 함께 봅니다.

## 통과 기준

- 실제 영수증 한 장의 `입력 → OCR → 구조화 → 검증 → 사람 승인 → Excel`을 설명할 수 있습니다.
- 같은 방식으로 견적서·신청서·거래명세서 PoC를 설계할 수 있습니다.
- 실제 문서로 작동하는 작은 프로토타입과 전체 정답 코드를 가지고 있습니다.
- 개인·회사 문서를 승인 없이 외부 클라우드에 올리지 않습니다.

## 참고 자료

공식 근거는 [과정 참고자료와 적용 범위](../docs/course_references.md)의 8교시 표에서 확인할 수 있습니다.
