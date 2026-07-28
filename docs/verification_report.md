# 2026 Document AI 교재 검증 보고서

검증일: 2026-07-28<br>
대상 브랜치: `document_ai_lecture_2026`<br>
최신 로컬 검증 환경: Python 3.14.3, macOS ARM64

## 결론

공지된 8개 교시 제목을 유지하면서 공개 한국 영수증 한 장을
`OCR → 공간 순서 복원 → 구조화 → 근거 있는 JSON → 검증 → 사람 수정·재검증
→ 승인 → Excel`로 연결했다. 모든 필수 실습은 Google Colab 노트북이며,
전 교시에 자기 답 빈칸·힌트·전체 정답이 있다.

최신 저장소 코드는 자동검증 기준 **GO**다. 다만 최종 커밋을 사용한 새 Colab과
Windows 녹화 PC 점검은 외부 사전 점검 항목으로 남긴다.

## 최신 자동 검증 결과

| 검증 | 결과 |
| --- | --- |
| Python 구문 검사 | 통과 |
| 단위·통합·Streamlit AppTest | 46개 통과 |
| Colab 준비 경로 독립 실행 | 8/8 통과 |
| Colab 2→3→4→7 순차 인계 | 5품목·승인 Excel·최종 앱 ZIP까지 통과 |
| Colab 단계 안내 | 코드 셀 65/65에 단계·행동·확인 결과·완료 안내 |
| 실제 PP-OCRv5 기록 회귀 | 44토큰 → 총액 76,000원·품목 5개 |
| 최종 앱 사람 수정 검증 | 총액 999 차단, 정상값 복구 후 승인·다운로드 |
| 교재 구조·로컬 링크 | 교재 8개·Colab 8개 통과 |
| Office 구조 | Excel·Word·PDF·PPT 4종 통과 |
| 실물형 합성 문서 사진 | 견적서·신청서·거래명세서 3종 통과 |
| PowerPoint 시각 검사 | 한글·원본 이미지 표시, 오버플로 0건 |
| Excel 시각 검사 | 3개 시트 렌더, 수식·통화 서식·가독성 확인 |
| 최종 Streamlit 서버 | health `ok`, 루트 HTTP 200 |
| Git 관리 대상 README | 루트 1개 |

검증 명령:

```bash
python -m compileall -q app.py src tests tools
python -m pytest -q
python tools/build_colab_notebooks.py
python tools/validate_colab_notebooks.py
python tools/validate_course_materials.py
python tools/validate_office_samples.py
git diff --check
```

## 교시별 실행 산출물

| 교시 | 최신 확인 산출물 |
| --- | --- |
| 1 | `lesson01_comparison_report.json` |
| 2 | `lesson02_ocr_outputs.zip` (`ocr_result.json`, `ocr_boxes.png`) |
| 3 | `clean_receipt.json` |
| 4 | `receipt.json`, `vlm_comparison.json` |
| 5 | `app_05.py` |
| 6 | `app_06.py` |
| 7 | `receipt_result.xlsx`, `final_document_ai_app.zip` |
| 8 | `poc_candidate_card.md`, `office_format_samples.zip` |

## 1교시 전면 교체 검수

기존의 기술명과 파일명 연결 실습은 학습 목표와 맞지 않아 폐기했다. 새 1교시는
한 장의 한국 영수증에서 실제 OCR 결과, 교육용 VLM 오류 초안, Document AI
검증, 사람 승인을 연속해서 경험하도록 다시 만들었다.

| 검수 항목 | 확인한 증거 | 결과 |
| --- | --- | --- |
| 주제 부합성 | OCR 글자·위치·신뢰도, VLM 구조 초안, Document AI 검증, IDP 승인 경험 | 통과 |
| 실제성 | 같은 영수증의 PP-OCRv5 LIVE 기록 44토큰과 오인식 두 곳 사용 | 통과 |
| 출처 정직성 | VLM 결과를 현재 모델 호출이 아닌 교육용 오류 삽입 초안으로 명시 | 통과 |
| 수강생 행동 | 원본 관찰·OCR 검토·VLM 검토·값 수정·승인·개념 선택의 TODO 6개 | 통과 |
| 입력 반영 | 수강생이 작성한 8개 시도가 최종 `learner_attempts`에 저장 | 통과 |
| 검증 경험 | 16,000원 초안에서 오류 3개 발견, 76,000원과 원문 근거 수정 뒤 통과 | 통과 |
| 시각 확인 | 실제 이미지 위 44개 사각형의 위치와 신뢰도 색상 정렬 확인 | 통과 |
| 실행 복구 | 원격 자료 실패 시 수동 업로드 안내, 자동 검증은 저장소 로컬 fixture 사용 | 통과 |
| 다음 교시 연결 | 1교시는 결과 비교, 2교시는 PP-OCRv5 실제 실행으로 역할 분리 | 통과 |

자동 검증은 20개 이상의 셀, TODO 6개, 기존 파일명 연결 코드 부재,
수정 전 실패·수정 후 성공, 사람 승인, 다섯 개 개념 정답을 별도로 검사한다.

## 실제 OCR 회귀가 필요한 이유

이전 LIVE 실행은 OCR 모델이 한 품목 행을 여러 토큰으로 반환했다. 토큰 배열을
단순 줄바꿈하면 합계와 품목의 관계가 깨져 앱이 열려도 `total=null`,
`items=[]`가 될 수 있었다.

최신 구현은 바운딩 박스의 y좌표로 같은 행을 묶고, 행 안에서 x좌표로
정렬한다. 실제 PP-OCRv5 LIVE 실행에서 보존한 44개 토큰을 테스트 fixture로
고정해 다음을 매번 검사한다.

```text
거래일: 2025-10-04
총액: 76000
품목: 5개
```

이 검사는 준비 텍스트 성공과 실제 OCR 토큰 성공을 서로 바꿔 말하지 않는다.

## OCR·VLM 결과의 계보

- `receipt.json`: 3교시 OCR 정제 텍스트를 규칙으로 구조화한 기준선
- `vlm_comparison.json`: 같은 공개 문서의 준비 VLM 구조 예시
- 준비 VLM 예시: `engine=not_executed`, 현재 실행 결과가 아니라는 안내 포함
- 실제 VLM: 승인된 강사 계정·비식별 샘플·예산 상한이 있을 때 한 번만 시연

따라서 정규식을 실행한 결과를 VLM 추론이라고 표시하지 않는다.

## 최종 앱에서 확인한 보호 경로

- 파일 없음, 허용하지 않은 확장자, 5MB 초과
- OCR 의존성·모델 오류와 명시적 준비 결과 선택
- 상호명·날짜·총액·품목 표 편집
- 사람의 수정값을 포함한 재검증
- 필수값·자료형·품목 계산·합계·원문 근거 오류
- 미승인 또는 오류 상태의 Excel 다운로드 차단
- 선행 공백·제어문자 뒤 수식 접두 문자 보호
- 승인자·승인 시각·수정 이유·처리 모드 기록
- 승인 Excel의 `검토_요약`, `품목`, `원문_근거` 3시트

## 과거 실제 Colab 기록과 최신 코드의 경계

2026-07-28 강사 PC의 로그인된 Colab에서는 PaddleOCR 3.7.0,
PP-OCRv5 Korean LIVE 44영역과 8개 교시 체크포인트를 실행했다. 이 기록은
공간 복원 결함을 찾는 근거로 사용했다.

그 뒤 최종 앱 수정·재검증, 전 교시 빈칸, ZIP 다운로드, Colab iframe,
Office 파일 재임베드가 추가됐다. 이 최신 커밋 전체를 새 Colab에서 다시
실행한 기록은 사전 점검표가 완료되기 전까지 **미검증**으로 유지한다.

## 남은 외부 검증

- 검증 기능·교재 커밋 `53ce1b1` 원격 push·SHA 일치 확인 완료
- 비로그인 Colab 8개 접근
- 최종 커밋으로 새 Colab CPU 세션 재실행
- 첫 PaddleOCR 모델 다운로드 시간 측정
- Windows 녹화 PC의 Chrome·iframe·다운로드·Office 한글 확인
- 20명 현장 네트워크와 복구 라인 확인

무료 Colab의 자원과 실행 시간은 보장하지 않는다. 공개·합성 샘플의 성공을
일반 정확도·비용 절감률로 확대하지 않으며, 실제 조직 적용 전 개인정보,
외부 전송, 접근권한, 보존·삭제 기준을 별도로 승인받아야 한다.
