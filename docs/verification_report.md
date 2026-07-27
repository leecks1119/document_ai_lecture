# 2026 Document AI 교재 검증 보고서

검증일: 2026-07-27  
대상 브랜치: `document_ai_lecture_2026`

## 결론

2026 과정의 활성 기술 스택을 `PaddleOCR 3.7 + PP-OCRv5 Korean`과 `PaddleOCR-VL 1.6`으로 교체했다. 1~8교시 교재와 Colab의 기본 mock 경로는 Python 3.12.11 환경에서 독립 실행됐다.

실제 OCR·VLM 모델 셀은 다운로드와 런타임 자원이 필요한 **선택 경로**다. 이번 로컬 검증에서 실제 추론까지 실행했다고 주장하지 않으며, 교육생의 필수 산출물은 준비된 동일 형식의 결과로 완성된다.

## 자동 검증 결과

| 검증 | 결과 |
| --- | --- |
| Python 구문 검사 | 통과 |
| 단위·통합 테스트 | 26개 통과 |
| Colab 기본 경로 독립 실행 | 8개 통과 |
| 교재 구조 검사 | 8개 통과 |
| 교육 도식 연결 | 16개 통과 |
| 로컬 파일 링크 | 통과 |
| 교시별 핵심 개념 | 정확히 3개 |
| 교시별 기본 실습 | 정확히 1개 |
| 교시별 주 산출물 | 정확히 1개 |
| 노트북 저장 출력 | 0개 |

사용한 명령:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python tools/validate_course_materials.py
.venv/bin/python tools/validate_colab_notebooks.py
.venv/bin/python -m compileall -q src app.py tools
```

## 기술 기준 확인

| 항목 | 2026 과정 기준 | 확인 내용 |
| --- | --- | --- |
| 일반 OCR | PaddleOCR 3.7.0 | 2026-06-11 공개 버전으로 고정 |
| 한국어 OCR | PP-OCRv5 Korean | `lang="korean"`, `ocr_version="PP-OCRv5"` |
| 문서 멀티모달 | PaddleOCR-VL 1.6 | 이미지·레이아웃을 Markdown과 블록으로 처리 |
| 업무 추출 | 명시적 변환 함수 | 모델 중간 결과를 바로 정답 JSON으로 취급하지 않음 |
| 최종 확정 | 검증 규칙 + 사람 검토 | 필수값·품목 합계 오류가 있으면 CSV 저장 차단 |

PP-OCRv6의 현재 공식 언어 범위에는 한국어가 없으므로 한국어 영수증은 PP-OCRv5를 사용한다. PaddleOCR-VL은 VLM 구성 요소만 단독 호출하지 않고 레이아웃 분석이 포함된 전체 파이프라인 예제로 작성했다.

근거:

- [PaddleOCR 3.7.0 패키지](https://pypi.org/project/paddleocr/)
- [PaddleOCR 3.x OCR 파이프라인](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [PP-OCRv5 다국어 인식](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
- [PaddleOCR-VL 1.6](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)
- [PaddleOCR-VL 파이프라인](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)

## 한국 실물 영수증 확인

1교시 입력은 Wikimedia Commons의 [2025년 태백 음식점 영수증](https://commons.wikimedia.org/wiki/File:Receipt_taebaek_restaurant_IMG_2614_modified.jpg)이다. Commons 파일 설명은 Public Domain(`PD-ineligible`)으로 표시한다.

저장소에는 원본을 그대로 넣지 않았다. 전화번호·내부 거래 식별 행을 추가로 가리고, 하단 결제 영역을 제외하고, 내장 메타데이터를 제거한 PNG만 포함했다. 원본·파생본 해시와 수정 내역은 `sample_docs/public_receipts/metadata.json`에서 확인한다. 해외 CORD 샘플은 1교시 대표 입력이 아니라 선택 비교 자료다.

## Colab 산출물 확인

| 교시 | 확인 산출물 |
| --- | --- |
| 1 | `technology_comparison.json` |
| 2 | `ocr_text.txt` |
| 3 | `clean_receipt.json` |
| 4 | `receipt.json` |
| 5 | `app_05.py` |
| 6 | `app_06.py` |
| 7 | `receipt.csv` |
| 8 | `business_application_card.md` |

2교시의 `RUN_PADDLEOCR`, 4교시의 `RUN_PADDLEOCR_VL`, 5교시의 `RUN_PUBLIC_DEMO`는 기본값이 모두 `False`다. 검증기는 선택 셀을 건너뛴 상태로 각 노트북을 새 임시 폴더에서 위에서 아래로 실행한다.

## 실패 경로 확인

자동 테스트에서 다음 입력을 거부하거나 오류 상태로 반환했다.

- 파일 없음
- 허용하지 않은 확장자
- 5MB 초과 파일
- 손상 PDF
- 암호 PDF
- 3페이지 초과 PDF
- 지원하지 않는 처리기 이름
- OCR 또는 VLM 의존성 없음
- 필수값 누락
- 품목 합계 불일치

오류 뒤 관련 없는 샘플을 자동 결과처럼 표시하지 않는다. 사용자가 `use_sample=True`를 명시적으로 선택해야 mock 경로가 실행된다.

## 운영 주의

- 실제 모델의 첫 실행은 패키지와 모델 다운로드에 영향을 받는다.
- PaddleOCR-VL은 OCR보다 메모리와 실행 시간이 더 필요하므로 Colab에서 선택 실행한다.
- 공개 한국 영수증 한 장과 합성 영수증의 결과를 일반적인 정확도 수치로 사용하지 않는다.
- Gradio 공개 주소에는 합성 문서만 사용한다.
- 실제 조직 적용 전 개인정보, 외부 전송, 접근권한, 보존·삭제 기준을 별도로 승인받는다.
