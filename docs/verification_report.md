# 2026 Document AI 교재 검증 보고서

검증일: 2026-07-27  
대상 브랜치: `document_ai_lecture_2026`
검증 환경: Python 3.12.11, macOS ARM64

## 결론

공지된 8교시 주제는 유지하면서 학습자 필수 실습을 Google Colab으로 통일했다. 1교시는 OCR·Multimodal AI·VLM·Document AI·IDP의 경계와 0~12 전체 참조 지도를 다루고, 2교시부터 실제 구현을 단계별로 완성한다.

웹앱은 Streamlit 1.60.0으로 구현했다. Colab에서는 공개 서버나 터널을 열지 않고 `streamlit.testing.v1.AppTest`로 검사한다. 최종 Excel은 검증 규칙을 통과하고 사람이 원본을 확인한 뒤에만 생성한다.

실제 OCR·VLM 모델 셀은 다운로드와 런타임 자원이 필요한 선택 경로다. 이번 자동 검증은 모든 오프라인·준비 결과 경로를 실행했으며 실제 모델 추론까지 실행했다고 주장하지 않는다.

## 자동 검증 결과

| 검증 | 결과 |
| --- | --- |
| Python 전체 구문 검사 | 통과 |
| 단위·통합·Streamlit AppTest | 29개 통과 |
| Colab 오프라인 경로 독립 실행 | 8개 통과 |
| 교재 구조 검사 | 8개 통과 |
| 로컬 파일·이미지 링크 | 통과 |
| 교시별 기본 실습 | 정확히 1개 |
| 교시별 주 산출물 | 정확히 1개 |
| 노트북 저장 출력 | 0개 |
| Git 관리 대상 README | 루트 1개 |

사용한 명령:

```bash
.venv/bin/python -m compileall -q .
.venv/bin/python -m pytest -q
.venv/bin/python tools/validate_course_materials.py
.venv/bin/python tools/validate_colab_notebooks.py
```

## 기술 기준

| 항목 | 과정 기준 | 확인 내용 |
| --- | --- | --- |
| 한국어 OCR | PaddleOCR 3.7 + PP-OCRv5 Korean | 2교시 실제 실행을 먼저 시도하고 3분 뒤 준비 결과로 복구 |
| 문서 멀티모달 | PaddleOCR-VL 1.6 | 강사 비식별 시연 1회와 준비된 중간 결과 사용 |
| 웹앱 | Streamlit 1.60.0 | 모든 학습자 코드는 Colab에서 만들고 AppTest로 검사 |
| 데이터 구조 | 명시적 JSON 스키마 | 원문·정제값·근거·검증 상태를 분리 |
| 최종 확정 | 검증 규칙 + 사람 승인 | 둘 중 하나라도 없으면 Excel 다운로드 차단 |
| 최종 파일 | `receipt_result.xlsx` | 검토 요약·품목·원문 시트 생성 |

근거:

- [Google Cloud Document AI 개요](https://cloud.google.com/document-ai/docs/overview)
- [AWS Intelligent Document Processing 설명](https://aws.amazon.com/what-is/intelligent-document-processing/)
- [PP-OCRv5 다국어 인식](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
- [PaddleOCR-VL 1.6](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)

## 한국 실물 영수증

1교시 입력은 Wikimedia Commons의 [2025년 태백 음식점 영수증](https://commons.wikimedia.org/wiki/File:Receipt_taebaek_restaurant_IMG_2614_modified.jpg)이다. Commons 파일 설명은 Public Domain(`PD-ineligible`)으로 표시한다.

저장소에는 전화번호·내부 거래 식별 영역을 추가로 가리고 메타데이터를 제거한 PNG만 포함한다. 원본·파생본 해시와 수정 내역은 `sample_docs/public_receipts/metadata.json`에 기록했다. 1교시의 금액과 근거 문구는 교재 제작자가 원본에서 확인한 교육용 준비 결과이며, 모델의 신뢰도나 좌표처럼 표시하지 않는다.

## Colab 산출물

| 교시 | 확인 산출물 |
| --- | --- |
| 1 | `receipt_pipeline_trace.json` |
| 2 | `ocr_result.json` |
| 3 | `clean_receipt.json` |
| 4 | `receipt.json` |
| 5 | `app_05.py` |
| 6 | `app_06.py` |
| 7 | `receipt_result.xlsx` |
| 8 | `poc_candidate_card.md` |

2교시의 `RUN_PADDLEOCR`와 4교시의 `RUN_PADDLEOCR_VL`은 기본값이 `False`다. 검증기는 선택 셀을 건너뛴 상태로 각 노트북을 새 임시 폴더에서 위에서 아래로 실행한다.

## 확인한 실패·보호 경로

- 파일 없음, 허용하지 않은 확장자, 5MB 초과 파일
- 손상·암호화·페이지 제한 초과 PDF
- 지원하지 않는 처리기 이름
- OCR 또는 VLM 의존성 없음
- 필수값 누락, 날짜·금액 형식 오류, 품목 합계 불일치
- 검증 통과 전 Excel 생성 차단
- 사람 승인 전 Excel 생성·다운로드 차단
- 스프레드시트 수식 접두 문자가 있는 추출 문자열 보호

오류 뒤 관련 없는 샘플을 자동 결과처럼 표시하지 않는다. 사용자가 준비 결과를 명시적으로 선택해야 하며 화면과 산출물에 사용 모드를 남긴다.

## 검증 범위의 한계

- 무료 Colab의 자원·GPU·사용 시간은 보장되지 않는다.
- 모델 최초 실행은 패키지와 가중치 다운로드 상태에 영향을 받는다.
- 공개 영수증 한 장과 합성 영수증 결과를 일반적인 정확도 수치로 사용하지 않는다.
- 실제 조직 적용 전 개인정보, 외부 전송, 접근권한, 보존·삭제 기준을 별도로 승인받아야 한다.
