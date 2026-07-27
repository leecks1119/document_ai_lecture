# 2026 Document AI 교재 검증 보고서

검증일: 2026-07-28<br>
대상 브랜치: `document_ai_lecture_2026`<br>
검증 환경: Python 3.12.11, macOS ARM64

## 결론

공지된 8교시 제목을 유지하면서 같은 공개 한국 영수증을 `OCR → 구조화 → 근거 있는 JSON → 검증 → 사람 승인 → Excel`로 연결했다. 수강생은 모든 필수 코드를 Google Colab에서 실행하며, 준비 결과는 `LIVE`와 혼동하지 않도록 `PREPARED_FALLBACK` 또는 `PREPARED REPLAY`로 표시한다.

자동 검증은 코드, 8개 노트북의 오프라인 복구 경로, 수강생 교재 링크, Office 4종의 내부 구조를 확인했다. Streamlit 앱은 실제 브라우저에서 준비 결과와 승인 뒤 Excel 다운로드까지 확인했다.

## 자동 검증 결과

| 검증 | 결과 |
| --- | --- |
| Python 구문 검사 | 통과 |
| 단위·통합·Streamlit AppTest | 41개 통과 |
| Colab 오프라인 경로 독립 실행 | 8개 통과 |
| 2→3→4→7 Colab 순차 인계 | 같은 공유 폴더에서 5품목·승인 Excel까지 통과 |
| 교재 구조·로컬 링크 | 교재 8개, Colab 8개 통과 |
| Office 구조 | Excel·Word·PDF·PPT 4종 통과 |
| 실물형 합성 문서 사진 | 견적서·신청서·거래명세서 3종 통과 |
| PowerPoint 범위 초과 | 없음 |
| Word·PDF·PowerPoint 렌더 | 각 1페이지·1슬라이드 확인 |
| Git 관리 대상 README | 루트 1개 |

사용한 명령:

```bash
.venv/bin/python -m compileall -q app.py src tests tools
.venv/bin/python -m pytest -q
.venv/bin/python tools/validate_course_materials.py
.venv/bin/python tools/validate_colab_notebooks.py
.venv/bin/python tools/validate_office_samples.py
git diff --check
```

## 종단간 산출물

| 교시 | 확인 산출물 |
| --- | --- |
| 1 | `receipt_pipeline_trace.json` |
| 2 | `ocr_result.json`, `ocr_boxes.png` |
| 3 | `clean_receipt.json` |
| 4 | `receipt.json` |
| 5 | `app_05.py` |
| 6 | `app_06.py` |
| 7 | `receipt_result.xlsx` |
| 8 | `poc_candidate_card.md` |

2교시는 Colab에서 실제 PaddleOCR 실행을 기본 시도로 삼는다. 자동 검증기는 네트워크와 모델 다운로드에 의존하지 않도록 강제로 준비 결과 경로를 실행한다. 각 노트북은 새 Colab 세션에서도 이어 갈 수 있도록 선행 산출물 다운로드·업로드 함수를 제공한다. 순차 검증에서는 같은 공유 폴더에서 `02→03→04→07`을 실행해 OCR 영역 10개, 품목 후보 5개, 영수증 품목 5개, `수제 돈가스`, 승인 Excel 생성을 확인했다.

## 실제 화면·시각 검증

- 실제 Streamlit 앱에서 `PREPARED REPLAY` 상태가 크게 보이는지 확인했다.
- 원본 대조 체크 전에는 Excel 다운로드가 없고 승인 뒤에만 나타나는지 확인했다.
- OCR과 VLM 도식에서 두 경로 사이의 순서 화살표를 제거했다.
- 검증 도식에서 `valid`, `warnings`, `errors`를 병렬 판정으로, 사람 승인을 별도 게이트로 표시했다.
- Word, PDF, PowerPoint는 렌더 이미지에서 한글, 표, 여백, 잘림을 확인했다.
- Excel은 Artifact Tool 렌더와 구조 검사로 3시트, 수식, 숫자 표시를 확인했다.
- 앱 업로더는 화면과 서버 양쪽에서 5MB 제한을 표시·강제한다.

## 확인한 보호 경로

- 파일 없음, 허용하지 않은 확장자, 5MB 초과
- 손상·암호화·페이지 제한 초과 PDF
- OCR 또는 VLM 의존성·모델 오류
- `LIVE_ERROR` 뒤 관련 없는 준비 결과 자동 대체 금지
- 필수값·자료형·품목 계산·합계·원문 근거 오류
- 미승인 Excel 생성·다운로드 차단
- 선행 공백·제어문자 뒤 수식 접두 문자 보호
- 승인자·승인 시각·수정 이유·처리 모드 기록

## 검증 범위의 한계

- `document_ai_lecture_2026` 원격 브랜치가 아직 없어 8개 Colab URL의 공개 접근은 현재 실패한다. 브랜치를 push하고 비로그인 시크릿 창에서 8개 링크를 확인하기 전까지 실제 교육 시작 판정은 `NO-GO`다.
- 공식 Colab 새 세션에서 PaddleOCR 모델 다운로드와 LIVE 추론은 강의 T-48시간 사전 점검 항목이다. 현재 저장소 자동 검증 결과와 구분한다.
- 무료 Colab의 자원·사용 시간·모델 다운로드 속도는 보장하지 않는다.
- 공개 샘플과 합성 문서의 성공을 일반 정확도 수치로 확대하지 않는다.
- 실제 조직 적용 전 개인정보, 외부 전송, 접근권한, 보존·삭제 기준을 별도 승인받아야 한다.
- 하루 강의의 가격 가치는 저장소만으로 보장할 수 없으며, 실제 완주율·평가 결과·업무 전이로 사후 확인해야 한다.

## 출시 판정

| 범위 | 판정 | 근거 |
| --- | --- | --- |
| 저장소 내부 교육자료 | **GO** | 최종 독립 감사 90/100, 내부 P0 0건, 내부 P1 0건 |
| 실제 공개·교육 시작 | **NO-GO** | 원격 브랜치·새 Colab LIVE·현장 20명 점검·운영시간 선택 미완료 |

출시 전에는 강사용 사전 점검표의 T-48 항목을 실제 실행 증거로 채우고, 배포 commit SHA와 8개 Colab URL의 비로그인 접근 결과를 기록한다.
