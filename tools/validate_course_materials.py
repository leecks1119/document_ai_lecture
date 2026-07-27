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
    "05_streamlit_basic.md",
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
    "## 7. 실습 결과 확인",
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


def section(markdown: str, start: str, end: str) -> str:
    """두 2단계 제목 사이의 본문을 반환한다."""

    return markdown.split(start, 1)[1].split(end, 1)[0]


def validate_lesson(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    limit = 360 if path.name.startswith("01_") else 230
    assert len(lines) <= limit, f"{path.name}: 초보자 교재 분량이 {limit}줄을 넘음"
    assert all(section in text for section in REQUIRED_SECTIONS), path.name
    assert any(
        term in text.lower() for term in ("mock", "준비 결과", "복구본")
    ), f"{path.name}: 완성 복구 경로 없음"
    assert "Open In Colab" in text, f"{path.name}: Colab 배지 없음"

    artifact_section = section(
        text,
        "## 2. 이번 교시의 결과물",
        "## 3. 시작하기 전에",
    )
    artifacts = re.findall(r"^- ", artifact_section, flags=re.MULTILINE)
    assert len(artifacts) == 1, f"{path.name}: 주 산출물은 정확히 1개"

    concept_section = section(text, "## 4. 핵심 개념", "## 5. 전체 실습 흐름")
    concepts = re.findall(r"^### 4\.\d+ ", concept_section, flags=re.MULTILINE)
    if path.name.startswith("01_"):
        assert 3 <= len(concepts) <= 6, f"{path.name}: 1교시 핵심 개념 범위 오류"
    else:
        assert len(concepts) == 3, f"{path.name}: 핵심 개념은 정확히 3개"

    practice_section = section(text, "## 6. 단계별 실습", "## 7. 실습 결과 확인")
    practices = re.findall(r"^### 실습 ", practice_section, flags=re.MULTILINE)
    assert len(practices) == 1, f"{path.name}: 기본 실습은 정확히 1개"

    image_links = [
        link
        for link in re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
        if not link.startswith("http")
    ]
    expected_minimum = 3 if path.name.startswith("01_") else 0
    assert len(image_links) >= expected_minimum, (
        f"{path.name}: 1교시는 교육 이미지가 최소 3개 필요"
    )

    for link in local_links(text):
        target = (path.parent / link).resolve()
        assert target.exists(), f"{path.name}: 링크 대상 없음 {link}"

    lesson_number = path.name[:2]
    matching_notebooks = list(COLAB.glob(f"{lesson_number}_*.ipynb"))
    assert len(matching_notebooks) == 1, f"{path.name}: Colab 노트북 연결 오류"

    banned = ["EasyOCR", "Gradio", "Codex", "실제 주민등록번호"]
    for term in banned:
        assert term not in text, f"{path.name}: 필수 범위 밖 용어 {term}"

    if lesson_number in {"02", "04", "06"}:
        assert "PaddleOCR" in text, f"{path.name}: 최신 처리 경로 설명 없음"


def validate_repository_links() -> None:
    for relative_path in [
        "README.md",
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
    print("검증 완료: 교재 8개, Colab 8개, 로컬 링크")


if __name__ == "__main__":
    main()
