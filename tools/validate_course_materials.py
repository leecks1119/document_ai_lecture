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

EXPECTED_OUTPUTS = {
    "01": ["lesson01_comparison_report.json"],
    "02": ["ocr_result.json", "ocr_boxes.png", "lesson02_ocr_outputs.zip"],
    "03": ["clean_receipt.json"],
    "04": ["receipt.json", "vlm_comparison.json"],
    "05": ["app_05.py"],
    "06": ["app_06.py"],
    "07": ["receipt_result.xlsx", "final_document_ai_app.zip"],
    "08": [
        "poc_candidate_card.md",
        "office_format_samples.zip",
        "business_document_code_examples.zip",
    ],
}

INTERNAL_QA_PHRASES = [
    "CHECKPOINT",
    "AppTest",
    "fixture_type",
    "course maintainer",
    "RECORDED LIVE REGRESSION",
    "회귀 검사",
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
    assert "## 직접 해보기" in text, f"{path.name}: 수강생 실습 안내 없음"
    assert "## 참고자료" in text, f"{path.name}: 참고자료 안내 없음"
    assert "Colab 실습 열기" in text, f"{path.name}: Colab 링크 없음"
    assert text.count("이번 시간의 도착점") == 1, (
        f"{path.name}: '이번 시간의 도착점' 안내는 정확히 한 번이어야 함"
    )

    lesson_number = path.name[:2]
    for output in EXPECTED_OUTPUTS[lesson_number]:
        assert output in text, f"{path.name}: 실습 산출물 설명 없음 {output}"

    closing_sections = [
        "## 이번 시간의 정리",
        "## 결과를 해석해 봅시다",
        "## 실습 결과",
        "## 과정을 마치며",
    ]
    assert any(section in text for section in closing_sections), (
        f"{path.name}: 학습 결과를 해석하거나 정리하는 설명 없음"
    )

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

    for phrase in INTERNAL_QA_PHRASES:
        assert phrase not in text, f"{path.name}: 내부 검증 표현이 학생 교재에 남음 {phrase}"

    if lesson_number in {"02", "04", "06"}:
        assert "PaddleOCR" in text, f"{path.name}: 최신 처리 경로 설명 없음"


def validate_repository_links() -> None:
    documentation_paths = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "lessons").glob("*.md")),
        *sorted((ROOT / "instructor").rglob("*.md")),
        *sorted((ROOT / "prompts").glob("*.md")),
    ]
    for path in documentation_paths:
        relative_path = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for link in local_links(text):
            target = (path.parent / link).resolve()
            assert target.exists(), f"{relative_path}: 링크 대상 없음 {link}"

    reference_text = (ROOT / "docs/course_references.md").read_text(encoding="utf-8")
    official_links = set(re.findall(r"https://[^)\s]+", reference_text))
    assert len(official_links) >= 10, "과정 참고자료는 공식·1차 출처 10개 이상 필요"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in INTERNAL_QA_PHRASES:
        assert phrase not in readme, f"README.md: 내부 검증 표현이 수강생 안내에 남음 {phrase}"

    lesson_specific_checks = {
        "02_ocr_basic.md": (
            "PaddleOCR 3.7.0",
            'lang="korean"',
            'ocr_version="PP-OCRv5"',
        ),
        "05_streamlit_basic.md": ("공개 영수증 결과 확인",),
        "06_ocr_ai_integration.md": (
            "공개 샘플 준비 결과",
            "업로드 파일 LIVE 처리",
        ),
    }
    for lesson_name, phrases in lesson_specific_checks.items():
        lesson_text = (LESSONS / lesson_name).read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase in lesson_text, (
                f"{lesson_name}: 실제 실습 문구가 교재에 없음 {phrase}"
            )


def validate_instructor_notes() -> None:
    note_paths = sorted((ROOT / "instructor" / "lesson_notes").glob("[0-9][0-9]_*.md"))
    assert len(note_paths) == 8, "교시별 강사 노트는 8개여야 함"
    for path in note_paths:
        text = path.read_text(encoding="utf-8")
        for heading in ("## 전달할 한 문장", "## 60분 운영", "## 종료 조건"):
            assert heading in text, f"{path.name}: 강사 운영 항목 없음 {heading}"

        time_ranges = [
            (int(start), int(end))
            for start, end in re.findall(
                r"^\|\s*(\d+)~(\d+)\s*\|",
                text,
                flags=re.MULTILINE,
            )
        ]
        assert time_ranges, f"{path.name}: 60분 운영 시간표 없음"
        assert time_ranges[0][0] == 0 and time_ranges[-1][1] == 60, (
            f"{path.name}: 운영 시간표가 0분부터 60분까지 이어지지 않음"
        )
        assert all(
            current[1] == following[0]
            for current, following in zip(time_ranges, time_ranges[1:])
        ), f"{path.name}: 운영 시간표에 공백 또는 중복이 있음"


def main() -> None:
    for filename in LESSON_FILES:
        validate_lesson(LESSONS / filename)
        print("OK:", filename)

    validate_repository_links()
    validate_instructor_notes()
    print("검증 완료: 교재 8개, Colab 8개, 로컬 링크")


if __name__ == "__main__":
    main()
