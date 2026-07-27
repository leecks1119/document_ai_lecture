# Document AI 강의 교재

이 폴더는 Notion에 등록할 8교시 교육생용 강의 교재를 관리한다.

강의 운영 문서와 환경 설정 문서는 `docs/`에서 관리하고, 교육생이 수업 중 읽고 따라 할 본문은 이 폴더에서 관리한다.

## 교재 목록

| 교시 | 교재 | 주제 | 작성 상태 |
| --- | --- | --- | --- |
| 1교시 | [01_document_ai_overview.md](01_document_ai_overview.md) | OCR보다 먼저 정할 것 | 본문·Colab mock 검증 |
| 2교시 | [02_ocr_basic.md](02_ocr_basic.md) | OCR 결과를 눈으로 확인하기 | 본문·Colab mock 검증 |
| 3교시 | [03_document_structure.md](03_document_structure.md) | OCR 초안을 정돈된 데이터로 | 본문·Colab mock 검증 |
| 4교시 | [04_genai_extraction.md](04_genai_extraction.md) | 필요한 값만 JSON으로 | 본문·Colab mock 검증 |
| 5교시 | [05_gradio_basic.md](05_gradio_basic.md) | Python 함수에 화면 붙이기 | 본문·Colab mock 검증 |
| 6교시 | [06_ocr_ai_integration.md](06_ocr_ai_integration.md) | 작은 함수들을 한 줄로 연결하기 | 본문·Colab mock 검증 |
| 7교시 | [07_validation_export.md](07_validation_export.md) | 틀린 값을 걸러 CSV로 | 본문·Colab mock 검증 |
| 8교시 | [08_business_application.md](08_business_application.md) | 자동화의 마지막은 사람 확인 | 본문·Colab mock 검증 |

전체 과정의 목표와 교시별 산출물은 [교육 커리큘럼](../docs/curriculum.md)을 기준으로 한다.

## 파일 구성

```text
lessons/
├── README.md
├── _template.md
├── 01_document_ai_overview.md
├── 02_ocr_basic.md
├── 03_document_structure.md
├── 04_genai_extraction.md
├── 05_gradio_basic.md
├── 06_ocr_ai_integration.md
├── 07_validation_export.md
├── 08_business_application.md
└── assets/
    ├── README.md
    ├── 01/
    ├── 02/
    ├── 03/
    ├── 04/
    ├── 05/
    ├── 06/
    ├── 07/
    └── 08/
```

## 작성 원칙

- 각 교재는 `_template.md`의 제목과 섹션 순서를 따른다.
- Notion에서 읽기 쉽도록 제목은 `H1`부터 `H3`까지만 사용한다.
- 설명 다음에 예제와 실습을 배치하고, 긴 코드 블록은 작은 단계로 나눈다.
- 모든 실습에는 기대 결과와 완료 체크리스트를 포함한다.
- Codex 프롬프트는 목표, 맥락, 제약조건, 완료 기준을 구분해 작성한다.
- OCR 또는 API 사용이 불가능한 환경을 위한 mock 실습 경로를 함께 제공한다.
- 실제 API 키와 개인정보가 포함된 문서나 화면 이미지는 사용하지 않는다.
- 교시별 이미지는 `assets/{교시 번호}/`에 저장하고 상대 경로로 연결한다.

## Notion 등록 원칙

- Markdown 파일 하나를 Notion의 교시 페이지 하나로 등록한다.
- 교재 파일명 앞의 번호와 Notion 페이지의 교시 번호를 동일하게 유지한다.
- 모듈 페이지 아래에 각 교시 페이지를 배치한다.
- 이미지와 첨부 파일은 해당 교시의 `assets/{교시 번호}/`에서 가져온다.
- 저장소의 교재를 원본으로 유지하고, 수정 사항은 저장소에 먼저 반영한다.
