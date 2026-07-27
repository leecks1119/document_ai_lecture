# Google Colab 실습 노트북

이 폴더의 8개 노트북은 각각 새 런타임에서 독립적으로 실행할 수 있다. 이전 교시의 파일이 없어도 내장 합성 데이터로 시작한다.

| 교시 | 노트북 | 기본 산출물 |
| --- | --- | --- |
| 1 | [01_document_ai_overview.ipynb](01_document_ai_overview.ipynb) | `field_spec.json` |
| 2 | [02_ocr_basic.ipynb](02_ocr_basic.ipynb) | `ocr_text.txt` |
| 3 | [03_document_structure.ipynb](03_document_structure.ipynb) | `clean_receipt.json` |
| 4 | [04_genai_extraction.ipynb](04_genai_extraction.ipynb) | `receipt.json` |
| 5 | [05_gradio_basic.ipynb](05_gradio_basic.ipynb) | `app_05.py` |
| 6 | [06_ocr_ai_integration.ipynb](06_ocr_ai_integration.ipynb) | `app_06.py` |
| 7 | [07_validation_export.ipynb](07_validation_export.ipynb) | `receipt.csv` |
| 8 | [08_business_application.ipynb](08_business_application.ipynb) | `business_application_card.md` |

## 실행 원칙

- **런타임 → 런타임 유형 변경**에서 Python 3 런타임을 사용한다.
- 교재 검증 기준은 Colab `2026.04` 고정 런타임과 Python 3.12.x다.
- 필수 셀은 API 키와 OCR 모델 다운로드가 필요 없다.
- EasyOCR는 기본값이 `False`인 선택 셀이다. 생성형 AI는 API 연결 전 준비사항만 선택적으로 확인한다.
- Gradio 공개 공유 실행도 기본값이 `False`다.
- 모든 결과물은 현재 세션의 `course_outputs/`에 생성되므로 필요한 파일은 수업 중 다운로드한다.

## 재생성과 검증

노트북은 다음 스크립트로 재생성한다.

```bash
python tools/build_colab_notebooks.py
```

Python 3.12 환경에서 구조와 필수 mock 경로를 셀 순서대로 검사한다.

```bash
python tools/validate_colab_notebooks.py
```

검증기는 다음을 확인한다.

- 노트북 8개와 Colab 메타데이터
- 중복되지 않는 고정 셀 ID
- 저장된 실행 결과와 오류가 없는 깨끗한 배포본
- 선택 OCR·API 준비 확인·공유 실행의 기본값이 `False`
- 모든 코드 셀의 순차 실행
- 교시별 지정 산출물 생성
