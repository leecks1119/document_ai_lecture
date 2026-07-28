"""Colab 노트북의 준비 경로와 공개된 승인 시나리오를 실행 검증한다."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"

EXPECTED_ARTIFACTS = {
    "01_document_ai_overview.ipynb": "lesson01_comparison_report.json",
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
    # 생성 앱 전체 소스처럼 긴 문자열 안의 임의 문자열은 제외하고,
    # 사람이 읽는 짧은 코드 줄에서 도구명 잔존 여부를 검사한다.
    visible_source = "\n".join(
        line for line in source.splitlines() if len(line) < 500
    )
    assert "Google Colab도 외부 클라우드" in source, path
    assert "CHECKPOINT 1/1 PASS" in source, path
    assert "TODO" in source, f"{path}: 학습자 빈칸이 없습니다."
    assert "전체 정답" in source, f"{path}: 공개 정답이 없습니다."
    assert "easyocr" not in source.lower(), path
    assert "gradio" not in source.lower(), path
    assert "codex" not in visible_source.lower(), path
    assert "base64.b64decode" not in source, path
    assert "FALLBACK_IMAGE_BASE64" not in source, path
    assert "GOLDEN_IMAGE_BASE64" not in source, path
    longest_code_line = max(
        len(line)
        for cell in code_cells
        for line in cell["source"].splitlines()
    )
    assert longest_code_line < 500, (
        f"{path}: unreadable code line ({longest_code_line} chars)"
    )
    if path.name == "01_document_ai_overview.ipynb":
        assert len(notebook["cells"]) >= 20
        assert "RECORDED_PP_OCRV5_TOKENS" in source
        assert "VLM_DRAFT_WITH_ERROR" in source
        assert "MY_OCR_REVIEW" in source
        assert "MY_CORRECTED_TOTAL" in source
        assert "validate_candidate" in source
        assert "MY_CONCEPT_CHOICES" in source
        assert "멀티모달 AI" in source
        assert "learner_attempts" in source
        assert source.count("# TODO") >= 6
        assert "16000" in source and "76000" in source
        assert "my_role_map" not in source
        assert "ANSWER_ROLE_MAP" not in source
        assert "receipt_pipeline_trace" not in source
        assert "taebaek_restaurant_2025_redacted.png" in source
        assert "ppocrv5_live_receipt_tokens.json" in source
        assert "files.upload()" in source
    if path.name == "02_ocr_basic.ipynb":
        assert "RUN_LIVE_OCR = not VALIDATION_MODE" in source
        assert 'lang="korean"' in source
        assert 'ocr_version="PP-OCRv5"' in source
        assert "receipt_ocr_fallback.json" in source
    if path.name == "04_genai_extraction.ipynb":
        assert "evidence" in source and "provenance" in source
        assert '"engine": "not_executed"' in source
        assert "현재 실행에서 VLM을 호출한 결과가 아닙니다." in source
    if path.name == "05_streamlit_basic.ipynb":
        assert "uploaded.getvalue()" in source
        assert "serve_kernel_port_as_iframe" in source
    if path.name == "06_ocr_ai_integration.ipynb":
        assert "run_live_ocr" in source
        assert "RECORDED LIVE REGRESSION PASS" in source
        assert "serve_kernel_port_as_iframe" in source
        assert "PREPARED_FALLBACK" in source
        assert "LIVE_ERROR" in source
        assert "finally:" in source and "unlink(missing_ok=True)" in source
        assert "ppocrv5_live_receipt_tokens.json" in source
    if path.name == "07_validation_export.ipynb":
        assert "DEFAULT_BLOCKED PASS" in source
        assert "REVIEWED_APPROVED PASS" in source
        assert "FINAL APP PASS" in source
        assert "final_document_ai_app" in source
        assert "make_archive" in source
        assert "st.data_editor" in source
        assert "serve_kernel_port_as_iframe" in source
        assert "REVIEW_RECORD" in source
        assert "검토_요약\", \"품목\", \"원문_근거" in source
    if path.name == "08_business_application.ipynb":
        for document in ("quotation", "application", "transaction_statement"):
            assert document in source
        assert "office_format_samples.zip" in source
        assert "candidate = None" in source
        assert "score = {" in source
        assert "sample_docs/formats/quotation.xlsx" in source
        assert "sample_docs/extensions/quotation_photo.png" in source


def execute_prepared_path(path: Path, notebook: dict) -> None:
    namespace = {"__name__": "__notebook_validation__"}
    with tempfile.TemporaryDirectory(prefix=f"{path.stem}_") as temp_dir:
        previous = Path.cwd()
        previous_flag = os.environ.get("COURSE_VALIDATE_PREPARED")
        previous_asset_root = os.environ.get("COURSE_LOCAL_ASSET_ROOT")
        os.environ["COURSE_VALIDATE_PREPARED"] = "1"
        os.environ["COURSE_LOCAL_ASSET_ROOT"] = str(ROOT)
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
            if path.name == "01_document_ai_overview.ipynb":
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                assert payload["ocr"]["token_count"] == 44
                assert payload["vlm"]["model_called_in_this_notebook"] is False
                assert payload["document_ai"]["before_validation"]["valid"] is False
                assert payload["document_ai"]["after_validation"]["valid"] is True
                assert payload["document_ai"]["corrected_total"] == 76000
                assert payload["idp"]["human_review_decision"] == "APPROVED"
                assert len(payload["concept_answers"]) == 5
                assert len(payload["learner_attempts"]) == 8
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
            if previous_asset_root is None:
                os.environ.pop("COURSE_LOCAL_ASSET_ROOT", None)
            else:
                os.environ["COURSE_LOCAL_ASSET_ROOT"] = previous_asset_root


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
        previous_asset_root = os.environ.get("COURSE_LOCAL_ASSET_ROOT")
        os.environ["COURSE_VALIDATE_PREPARED"] = "1"
        os.environ["COURSE_LOCAL_ASSET_ROOT"] = str(ROOT)
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
            if previous_asset_root is None:
                os.environ.pop("COURSE_LOCAL_ASSET_ROOT", None)
            else:
                os.environ["COURSE_LOCAL_ASSET_ROOT"] = previous_asset_root


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
