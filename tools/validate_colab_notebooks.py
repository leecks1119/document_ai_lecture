"""Colab 노트북의 준비 경로와 공개된 승인 시나리오를 실행 검증한다."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"

EXPECTED_ARTIFACTS = {
    "01_document_ai_overview.ipynb": "receipt_pipeline_trace.json",
    "02_ocr_basic.ipynb": "ocr_result.json",
    "03_document_structure.ipynb": "clean_receipt.json",
    "04_genai_extraction.ipynb": "receipt.json",
    "05_streamlit_basic.ipynb": "app_05.py",
    "06_ocr_ai_integration.ipynb": "app_06.py",
    "07_validation_export.ipynb": "receipt_result.xlsx",
    "08_business_application.ipynb": "poc_candidate_card.md",
}


def validate_structure(path: Path, notebook: dict) -> None:
    assert notebook["nbformat"] == 4, path
    assert notebook["metadata"]["colab"]["name"] == path.name, path
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)), f"{path}: duplicate cell ids"
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells, f"{path}: no code cells"
    assert all(cell["outputs"] == [] for cell in code_cells), path

    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "Google Colab도 외부 클라우드" in source, path
    assert "CHECKPOINT 1/1 PASS" in source, path
    assert "easyocr" not in source.lower(), path
    assert "gradio" not in source.lower(), path
    assert "codex" not in source.lower(), path
    if path.name == "02_ocr_basic.ipynb":
        assert "RUN_LIVE_OCR = not VALIDATION_MODE" in source
        assert 'lang="korean"' in source
        assert 'ocr_version="PP-OCRv5"' in source
    if path.name == "04_genai_extraction.ipynb":
        assert "evidence" in source and "provenance" in source
        assert '"engine": "not_executed"' in source
        assert "현재 실행에서 VLM을 호출한 결과가 아닙니다." in source
    if path.name == "05_streamlit_basic.ipynb":
        assert "uploaded.getvalue()" in source
    if path.name == "06_ocr_ai_integration.ipynb":
        assert "run_live_ocr" in source
        assert "PREPARED_FALLBACK" in source
        assert "LIVE_ERROR" in source
        assert "finally:" in source and "unlink(missing_ok=True)" in source
    if path.name == "07_validation_export.ipynb":
        assert "DEFAULT_BLOCKED PASS" in source
        assert "REVIEWED_APPROVED PASS" in source
        assert "REVIEW_RECORD" in source
        assert "검토_요약\", \"품목\", \"원문_근거" in source
    if path.name == "08_business_application.ipynb":
        for document in ("quotation", "application", "transaction_statement"):
            assert document in source


def execute_prepared_path(path: Path, notebook: dict) -> None:
    namespace = {"__name__": "__notebook_validation__"}
    with tempfile.TemporaryDirectory(prefix=f"{path.stem}_") as temp_dir:
        previous = Path.cwd()
        previous_flag = os.environ.get("COURSE_VALIDATE_PREPARED")
        os.environ["COURSE_VALIDATE_PREPARED"] = "1"
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
            if path.name == "02_ocr_basic.ipynb":
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                assert payload["source_mode"] == "PREPARED_FALLBACK"
            if path.name == "07_validation_export.ipynb":
                assert not Path("course_outputs/pending_review.xlsx").exists()
                assert namespace["PENDING_REVIEW"]["decision"] == "PENDING"
                assert namespace["REVIEW_RECORD"]["decision"] == "APPROVED"
        finally:
            os.chdir(previous)
            if previous_flag is None:
                os.environ.pop("COURSE_VALIDATE_PREPARED", None)
            else:
                os.environ["COURSE_VALIDATE_PREPARED"] = previous_flag


def execute_sequential_handoff(paths: list[Path]) -> None:
    """02→03→04→07을 같은 산출물 폴더에서 새 노트북처럼 이어 실행한다."""

    selected = [
        path
        for lesson in ("02_", "03_", "04_", "07_")
        for path in paths
        if path.name.startswith(lesson)
    ]
    with tempfile.TemporaryDirectory(prefix="sequential_handoff_") as temp_dir:
        previous = Path.cwd()
        previous_flag = os.environ.get("COURSE_VALIDATE_PREPARED")
        os.environ["COURSE_VALIDATE_PREPARED"] = "1"
        os.chdir(temp_dir)
        try:
            for path in selected:
                notebook = json.loads(path.read_text(encoding="utf-8"))
                namespace = {"__name__": "__sequential_validation__"}
                for index, cell in enumerate(notebook["cells"], start=1):
                    if cell["cell_type"] != "code":
                        continue
                    try:
                        exec(
                            compile(
                                cell["source"],
                                f"{path.name}:sequential-cell-{index}",
                                "exec",
                            ),
                            namespace,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"순차 handoff에서 {path.name}의 {index}번째 셀 실행 실패"
                        ) from exc

            output_dir = Path("course_outputs")
            ocr_payload = json.loads(
                (output_dir / "ocr_result.json").read_text(encoding="utf-8")
            )
            assert len(ocr_payload["items"]) >= 10
            clean_payload = json.loads(
                (output_dir / "clean_receipt.json").read_text(encoding="utf-8")
            )
            assert len(clean_payload["groups"]["items"]) == 5
            receipt_payload = json.loads(
                (output_dir / "receipt.json").read_text(encoding="utf-8")
            )
            assert len(receipt_payload["items"]) == 5
            assert receipt_payload["items"][2]["name"] == "수제 돈가스"
            assert (output_dir / "receipt_result.xlsx").is_file()
        finally:
            os.chdir(previous)
            if previous_flag is None:
                os.environ.pop("COURSE_VALIDATE_PREPARED", None)
            else:
                os.environ["COURSE_VALIDATE_PREPARED"] = previous_flag


def main() -> None:
    paths = sorted(COLAB_DIR.glob("*.ipynb"))
    assert len(paths) == 8, f"expected 8 notebooks, found {len(paths)}"
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        validate_structure(path, notebook)
        execute_prepared_path(path, notebook)
        print("OK:", path.name)
    execute_sequential_handoff(paths)
    print("OK: 02→03→04→07 동일 폴더 순차 handoff")
    print(
        f"검증 완료: {len(paths)}개 노트북 · 독립 복구 경로 · "
        "공개 승인 경로 · 순차 handoff"
    )


if __name__ == "__main__":
    main()
