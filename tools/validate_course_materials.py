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
    "## 60분 뒤 남길 것",
    "## 개념 10%",
    "## 실습 90%",
    "## 통과 기준",
    "## 참고 자료",
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

    limit = 360 if path.name.startswith("01_") else 230
    assert len(lines) <= limit, f"{path.name}: 초보자 교재 분량이 {limit}줄을 넘음"
    assert all(section in text for section in REQUIRED_SECTIONS), path.name
    assert "Colab 실습 열기" in text, f"{path.name}: Colab 링크 없음"
    assert "course_outputs/" in text, f"{path.name}: 산출물 경로 없음"
    assert "이번 교시의 한 문장" in text, f"{path.name}: 핵심 메시지 없음"

    image_links = [
        link
        for link in re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
        if not link.startswith("http")
    ]
    expected_minimum = 3 if path.name.startswith("01_") else 2
    assert len(image_links) >= expected_minimum, (
        f"{path.name}: 실제 문서·화면·도식 이미지가 부족함"
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

    instructor_only_phrases = [
        "강사가",
        "교재 제작자",
        "60분 운영표",
        "RUN_PADDLEOCR_VL",
    ]
    for phrase in instructor_only_phrases:
        assert phrase not in text, f"{path.name}: 강사용 표현이 학생 교재에 남음 {phrase}"

    if lesson_number in {"02", "04", "06"}:
        assert "PaddleOCR" in text, f"{path.name}: 최신 처리 경로 설명 없음"


def validate_repository_links() -> None:
    for relative_path in [
        "README.md",
        "docs/environment.md",
        "docs/troubleshooting.md",
        "docs/course_references.md",
        "instructor/course_operation.md",
        "instructor/environment_and_models.md",
        "instructor/recovery_and_safety.md",
    ]:
        path = ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        for link in local_links(text):
            target = (path.parent / link).resolve()
            assert target.exists(), f"{relative_path}: 링크 대상 없음 {link}"

    reference_text = (ROOT / "docs/course_references.md").read_text(encoding="utf-8")
    official_links = set(re.findall(r"https://[^)\s]+", reference_text))
    assert len(official_links) >= 10, "과정 참고자료는 공식·1차 출처 10개 이상 필요"


def main() -> None:
    for filename in LESSON_FILES:
        validate_lesson(LESSONS / filename)
        print("OK:", filename)

    validate_repository_links()
    print("검증 완료: 교재 8개, Colab 8개, 로컬 링크")


if __name__ == "__main__":
    main()
