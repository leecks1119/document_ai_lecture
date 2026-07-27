"""6~8교시: 작은 처리 함수를 명시적인 기본·mock 경로로 연결한다."""

from __future__ import annotations

from pathlib import Path

from .export import receipt_to_csv_bytes
from .extract import mock_extract
from .ocr import extract_with_easyocr, load_mock_ocr, ocr_text_from_result
from .sample_data import (
    MISSING_STORE_RECEIPT,
    SAMPLE_OCR_TEXT,
    WRONG_TOTAL_RECEIPT,
)
from .validate import validate_receipt


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
MAX_FILE_SIZE = 5 * 1024 * 1024


def validate_upload(file_path: str | Path | None) -> list[str]:
    """수업용 업로드 정책을 확인하고 오류 목록을 반환한다."""

    if not file_path:
        return ["파일을 선택하세요."]
    path = Path(file_path)
    if not path.is_file():
        return ["업로드 파일을 찾을 수 없습니다."]
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return ["PNG, JPEG, PDF 파일만 사용할 수 있습니다."]
    if path.stat().st_size > MAX_FILE_SIZE:
        return ["수업에서는 5MB 이하 파일만 사용합니다."]
    return []


def process_document(
    file_path: str | Path | None = None,
    *,
    use_sample: bool = False,
) -> dict:
    """문서를 처리한다.

    `use_sample=True`는 사용자가 '샘플로 계속'을 명시적으로 선택한 경우다.
    업로드나 EasyOCR가 실패해도 자동으로 관련 없는 mock 결과를 반환하지 않는다.
    """

    if use_sample:
        ocr_text = SAMPLE_OCR_TEXT
        mode = (
            "MOCK OCR + MOCK 추출 — 업로드 문서를 읽지 않고 "
            "합성 샘플을 사용했습니다."
        )
    else:
        errors = validate_upload(file_path)
        if errors:
            return {
                "ok": False,
                "status": "입력 오류",
                "errors": errors,
                "can_continue_with_sample": True,
            }
        try:
            ocr_result = extract_with_easyocr(Path(file_path))
            ocr_text = ocr_text_from_result(ocr_result)
            mode = "LIVE EasyOCR + MOCK 추출 — JSON 구조화는 수업용 규칙입니다."
        except Exception as exc:
            return {
                "ok": False,
                "status": "OCR 오류",
                "errors": [str(exc)],
                "can_continue_with_sample": True,
            }

    extracted = mock_extract(ocr_text)
    extracted["source_mode"] = "mock_extraction"
    validation = validate_receipt(extracted)

    return {
        "ok": True,
        "status": mode,
        "ocr_text": ocr_text,
        "data": extracted,
        "validation": validation,
        "csv_bytes": receipt_to_csv_bytes(extracted),
    }


def run_smoke_test() -> dict[str, bool]:
    """8교시에서 한 번 호출해 정상·누락·합계 오류·mock 경로를 점검한다."""

    sample_result = process_document(use_sample=True)
    missing_result = validate_receipt(MISSING_STORE_RECEIPT)
    total_result = validate_receipt(WRONG_TOTAL_RECEIPT)
    return {
        "mock_path_works": bool(sample_result.get("ok")),
        "normal_result_is_valid": bool(
            sample_result.get("validation", {}).get("valid")
        ),
        "missing_required_is_blocked": not missing_result["valid"],
        "wrong_total_is_blocked": not total_result["valid"],
    }
