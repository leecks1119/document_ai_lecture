# 강사용 시각자료 사용 가이드

## 보여 주는 순서

실제 문서 → 모델이 본 위치 → 원문 텍스트 → JSON 근거 → 검증 오류 → 승인 → Excel 순서로 보여 준다. 추상 도식만 연속으로 두 장 이상 보여 주지 않는다.

## 교시별 주 시각 증거

| 교시 | 반드시 보여 줄 실제 증거 | 보조 도식 |
| --- | --- | --- |
| 1 | 공개 한국 영수증과 최종 Excel | 네 용어 관계, 0~12 지도 |
| 2 | 원본·bbox·저품질 이미지 | OCR 세 요소 |
| 3 | [실제 영수증 구조 주석](../lessons/assets/03/03_receipt_regions.png), raw/clean/change_log 실제 값 | 구조 지도 |
| 4 | evidence가 있는 실제 JSON | 스키마·근거·불확실성 |
| 5 | [실제 수업용 예제 앱 화면](../lessons/assets/screens/app_prepared_result.png)과 AppTest 통과 로그 | 최소 앱 구조 |
| 6 | OCR_ERROR와 [수업용 예제 상태 화면](../lessons/assets/screens/app_prepared_result.png) | 처리 경로 선택 |
| 7 | [잘못된 수정값 차단 화면](../lessons/assets/screens/app_validation_blocked.png), [승인 뒤 다운로드 화면](../lessons/assets/screens/app_approved_excel.png), Excel 3시트 | 수정·재검증·승인 게이트 |
| 8 | 문서 사진 3종과 Office 4파일 | PoC 카드 |

## 빔프로젝터 점검

- 본문 24pt 수준보다 작은 이미지는 확대해서 보여 준다.
- 영수증 전체와 합계 크롭을 번갈아 보여 준다.
- `직접 실행`, `COURSE_EXAMPLE`, `OCR_ERROR`는 말로만 설명하지 않고 화면 텍스트를 가리킨다.
- JSON은 전체를 보여 주지 말고 `value`, `evidence`, `source_mode` 세 곳만 확대한다.
- Excel은 시트 탭과 승인 기록을 한 화면에 모두 보이게 한다.

## 이미지 품질 기준

- 실제 문서가 들어 있는가?
- 이미지 한 장이 전달하는 메시지가 한 가지인가?
- 도식의 화살표가 고정 순서를 잘못 암시하지 않는가?
- valid·warnings·errors가 순차 단계가 아니라 병렬 판정으로 보이는가?
- 준비 결과가 실제 모델 결과처럼 보이지 않는가?
- 한글이 브라우저와 프로젝터에서 잘리지 않는가?
- 출처 또는 “교육용 합성 문서” 표시가 있는가?

## 피해야 할 화면

- 상자와 화살표만 있고 실제 문서가 없는 슬라이드
- 전체 코드를 한 화면에 축소한 슬라이드
- 근거 없이 “99% 정확” 같은 수치를 강조한 슬라이드
- 준비 결과를 직접 실행처럼 보이게 하는 화면
- 식별정보가 남은 개인 영수증
