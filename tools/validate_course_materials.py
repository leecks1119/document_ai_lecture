"""8교시 교재의 구조, 로컬 링크, 분량, 자산 연결을 검사한다."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "lessons"
COLAB = ROOT / "colab"

LESSON_FILES = [
    "01_document_ai_overview.md",
    "02_ocr_basic.md",
    "03_document_structure.md",
    "04_genai_extraction.md",
    "05_gradio_basic.md",
    "06_ocr_ai_integration.md",
    "07_validation_export.md",
    "08_business_application.md",
]

REQUIRED_SECTIONS = [
    "## 1. 학습 목표",
    "## 2. 이번 교시의 결과물",
    "## 3. 시작하기 전에",
    "## 4. 핵심 개념",
    "## 5. 전체 실습 흐름",
    "## 6. 단계별 실습",
    "## 7. Codex 활용",
    "## 8. 문제 해결",
    "## 9. 형성평가",
    "## 10. 핵심 요약",
    "## 11. 완료 체크리스트",
]


def local_links(markdown: str) -> list[str]:
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", markdown)
    return [
        link.split("#", 1)[0]
        for link in links
        if link
        and not link.startswith(("http://", "https://", "mailto:", "#"))
    ]


def validate_lesson(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert len(lines) <= 190, f"{path.name}: 초보자 교재 분량이 190줄을 넘음"
    assert all(section in text for section in REQUIRED_SECTIONS), path.name
    assert "mock" in text.lower(), f"{path.name}: mock 경로 없음"
    assert "Open In Colab" in text, f"{path.name}: Colab 배지 없음"

    image_links = [
        link
        for link in re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
        if not link.startswith("http")
    ]
    assert len(image_links) == 2, f"{path.name}: 교육 도식은 정확히 2개"

    for link in local_links(text):
        target = (path.parent / link).resolve()
        assert target.exists(), f"{path.name}: 링크 대상 없음 {link}"

    lesson_number = path.name[:2]
    matching_notebooks = list(COLAB.glob(f"{lesson_number}_*.ipynb"))
    assert len(matching_notebooks) == 1, f"{path.name}: Colab 노트북 연결 오류"

    banned = ["PaddleOCR", "Streamlit", "실제 주민등록번호"]
    for term in banned:
        assert term not in text, f"{path.name}: 필수 범위 밖 용어 {term}"


def validate_repository_links() -> None:
    for relative_path in [
        "README.md",
        "lessons/README.md",
        "colab/README.md",
        "docs/environment.md",
        "docs/troubleshooting.md",
        "docs/instructor_guide.md",
    ]:
        path = ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        for link in local_links(text):
            target = (path.parent / link).resolve()
            assert target.exists(), f"{relative_path}: 링크 대상 없음 {link}"


def main() -> None:
    for filename in LESSON_FILES:
        validate_lesson(LESSONS / filename)
        print("OK:", filename)

    validate_repository_links()
    print("검증 완료: 교재 8개, 도식 16개, Colab 8개, 로컬 링크")


if __name__ == "__main__":
    main()
