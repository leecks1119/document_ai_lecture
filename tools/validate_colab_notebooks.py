"""Colab 노트북의 구조와 기본 mock 경로를 Python에서 실행 검증한다."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"

EXPECTED_ARTIFACTS = {
    "01_document_ai_overview.ipynb": "technology_comparison.json",
    "02_ocr_basic.ipynb": "ocr_text.txt",
    "03_document_structure.ipynb": "clean_receipt.json",
    "04_genai_extraction.ipynb": "receipt.json",
    "05_gradio_basic.ipynb": "app_05.py",
    "06_ocr_ai_integration.ipynb": "app_06.py",
    "07_validation_export.ipynb": "receipt.csv",
    "08_business_application.ipynb": "business_application_card.md",
}


def validate_structure(path: Path, notebook: dict) -> None:
    assert notebook["nbformat"] == 4, path
    assert notebook["metadata"]["colab"]["name"] == path.name, path

    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)), f"{path}: duplicate cell ids"

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells, f"{path}: no code cells"
    for cell in code_cells:
        assert cell["execution_count"] is None, path
        assert cell["outputs"] == [], path

    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "RUN_PADDLEOCR = False" in source or path.name != "02_ocr_basic.ipynb"
    assert (
        "RUN_PADDLEOCR_VL = False" in source
        or path.name != "04_genai_extraction.ipynb"
    )
    assert "RUN_PUBLIC_DEMO = False" in source or path.name != "05_gradio_basic.ipynb"
    assert "OPENAI_API_KEY" not in source, f"{path}: do not embed key names in required path"
    assert "easyocr" not in source.lower(), f"{path}: EasyOCR must not appear"

    if path.name == "02_ocr_basic.ipynb":
        assert "PaddleOCR" in source
        assert 'lang="korean"' in source
    if path.name == "04_genai_extraction.ipynb":
        assert "PaddleOCRVL" in source
        assert 'pipeline_version="v1.6"' in source


def execute_mock_path(path: Path, notebook: dict) -> None:
    namespace = {"__name__": "__notebook_validation__"}
    with tempfile.TemporaryDirectory(prefix=f"{path.stem}_") as temp_dir:
        previous = Path.cwd()
        os.chdir(temp_dir)
        try:
            for index, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] != "code":
                    continue
                try:
                    exec(
                        compile(cell["source"], f"{path.name}:cell-{index}", "exec"),
                        namespace,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"{path.name}의 {index}번째 셀 실행 실패"
                    ) from exc

            artifact = Path("course_outputs") / EXPECTED_ARTIFACTS[path.name]
            assert artifact.is_file(), f"{path}: missing {artifact.name}"
            assert artifact.stat().st_size > 0, f"{path}: empty {artifact.name}"
        finally:
            os.chdir(previous)


def main() -> None:
    paths = sorted(COLAB_DIR.glob("*.ipynb"))
    assert len(paths) == 8, f"expected 8 notebooks, found {len(paths)}"

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        validate_structure(path, notebook)
        execute_mock_path(path, notebook)
        print("OK:", path.name)

    print(f"검증 완료: {len(paths)}개 노트북")


if __name__ == "__main__":
    main()
