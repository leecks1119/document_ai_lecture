# 과거 Document AI 강의자료 아카이브

이 폴더는 기존 Notion 강의자료를 2026년 개편 과정에서 참고하기 위한 구조화 아카이브다.

- 원본: [Document AI 강의](https://app.notion.com/p/281707c7ae7581beb748feca63ac4e16)
- 수집 기준일: 2026-07-27
- 수집 범위: 최상위 페이지 1개, 기존 교재 11개, 활용 아이디어 1개, 2026 리뉴얼안 1개
- 원본 변경 여부: 읽기 전용으로 조회했으며 Notion 원본은 수정하지 않음

이 아카이브는 Notion 페이지 전체를 그대로 복제한 백업이 아니라, 새 강의에 활용할 수 있도록 페이지별 내용·코드·시각 자료·주의점을 정리한 개편 참고 자료다.

## 폴더 구성

```text
legacy_materials/
├── README.md
├── source_repo_2025/
│   ├── notebooks/        # 과거 OCR 비교·앙상블 실습
│   └── docai_course/     # 과거 Python 패키지
├── notion_archive/
│   ├── 00_course_home.md
│   ├── 01_development_environment.md
│   ├── 02_document_ai_overview.md
│   ├── 03_paddleocr_basic.md
│   ├── 04_ocr_engine_comparison.md
│   ├── 05_confidence_visualization.md
│   ├── 06_image_preprocessing.md
│   ├── 07_ocr_ensemble.md
│   ├── 08_table_extraction.md
│   ├── 09_regex_information_extraction.md
│   ├── 10_cursor_toy_project.md
│   ├── 11_system_test_qna.md
│   ├── 12_business_ideas.md
│   └── 13_2026_renewal_plan.md
└── review/
    ├── reusable_materials_review.md
    ├── curriculum_mapping.md
    ├── fact_check_backlog.md
    └── superseded_2026_drafts/ # 기술 정정 전 작업 문서
```

## 판정 기준

| 판정 | 의미 |
| --- | --- |
| 재사용 | 개념이나 실습 구조를 큰 변경 없이 활용 가능 |
| 수정 후 재사용 | 방향은 유효하지만 도구, API, 예시 또는 설명을 최신화해야 함 |
| 참고만 | 8시간 입문 과정의 필수 범위를 벗어나 선택 자료로만 활용 |
| 제외 | 정확성, 보안, 난이도 또는 과정 목표 때문에 새 교재에 사용하지 않음 |
| 검증 필요 | 공식 자료나 실행 테스트로 확인하기 전에는 새 교재에 인용하지 않음 |

## 활용 원칙

- 새 교재의 기준은 `docs/curriculum.md`와 `lessons/`다.
- 이 폴더의 코드는 실행 가능한 최신 코드로 간주하지 않는다.
- `source_repo_2025/`에는 구형 OCR 엔진과 앙상블 코드가 포함될 수 있다.
- 수치, 제품 비교, 시장 순위, API 사용법은 2026-07-27 기준으로 다시 검증한다.
- 실명, 회사명, 연락처, 주민등록번호 등 개인정보성 샘플은 모두 가상 데이터로 교체한다.
- Cursor 중심 설명은 Codex 중심 워크플로로 다시 작성한다.
- OCR 설치가 실패해도 mock 결과로 후속 교시를 진행할 수 있어야 한다.
