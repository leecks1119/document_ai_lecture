"""6~8교시: 작은 처리 함수를 명시적인 기본·mock 경로로 연결한다."""

from __future__ import annotations

from pathlib import Path

from .export import receipt_to_xlsx_bytes
from .extract import extract_receipt_from_text
from .ocr import extract_with_paddleocr, ocr_text_from_result
from .sample_data import (
    GOLDEN_RECEIPT_OCR_TEXT,
    MISSING_STORE_RECEIPT,
    SAMPLE_OCR_TEXT,
    WRONG_TOTAL_RECEIPT,
)
from .validate import validate_receipt
from .vlm import parse_with_paddleocr_vl, vlm_text_from_result


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
    if path.suffix.lower() == ".pdf":
        try:
            import fitz

            with fitz.open(path) as document:
                if document.page_count != 1:
                    return ["필수 실습은 PDF 한 페이지만 처리합니다."]
        except ImportError:
            pass
    return []


def process_document(
    file_path: str | Path | None = None,
    *,
    use_sample: bool = False,
    processor: str = "ocr",
    human_approved: bool = False,
    review_record: dict | None = None,
) -> dict:
    """문서를 처리한다.

    `use_sample=True`는 사용자가 '샘플로 계속'을 명시적으로 선택한 경우다.
    업로드나 PaddleOCR/PaddleOCR-VL가 실패해도 자동으로 관련 없는 mock
    결과를 반환하지 않는다.
    """

    if use_sample:
        ocr_text = GOLDEN_RECEIPT_OCR_TEXT
        extraction_mode = "prepared_fixture_rule_extraction"
        mode = "PREPARED REPLAY — 공개 한국 영수증의 검수된 준비 결과"
        provenance = {
            "fixture_type": "human_verified_transcription_fixture",
            "input_file": "taebaek_restaurant_2025_redacted.png",
            "input_sha256": (
                "19227c7298a16ee69bef2d7bed65826b8a1cba5389375e4ae77d02005362641f"
            ),
            "engine": "not_executed",
            "engine_version": "not_applicable",
            "target_technology": "PaddleOCR Korean",
            "recorded_at": "2026-07-28",
            "reviewer": "course maintainer",
            "disclaimer": "현재 실행에서 모델을 호출한 결과가 아닙니다.",
        }
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
            if processor == "ocr":
                ocr_result = extract_with_paddleocr(Path(file_path))
                ocr_text = ocr_text_from_result(ocr_result)
                mode = (
                    "LIVE PaddleOCR 3.7 / PP-OCRv5 Korean + 규칙 추출"
                )
                extraction_mode = "live_ocr_rule_extraction"
                provenance = {
                    "fixture_type": "live_inference",
                    "input_file": Path(file_path).name,
                    "engine": "PaddleOCR",
                    "engine_version": "3.7 / PP-OCRv5 Korean",
                    "recorded_at": "",
                    "reviewer": "",
                    "disclaimer": "",
                }
            elif processor == "vlm":
                vlm_result = parse_with_paddleocr_vl(Path(file_path))
                ocr_text = vlm_text_from_result(vlm_result)
                mode = "LIVE PaddleOCR-VL 1.6 + 규칙 추출"
                extraction_mode = "live_vlm_rule_extraction"
                provenance = {
                    "fixture_type": "live_inference",
                    "input_file": Path(file_path).name,
                    "engine": "PaddleOCR-VL",
                    "engine_version": "1.6",
                    "recorded_at": "",
                    "reviewer": "",
                    "disclaimer": "",
                }
            else:
                return {
                    "ok": False,
                    "status": "처리 방식 오류",
                    "errors": ["processor는 'ocr' 또는 'vlm'이어야 합니다."],
                    "can_continue_with_sample": True,
                }
        except Exception as exc:
            return {
                "ok": False,
                "status": "문서 처리 오류",
                "errors": [str(exc)],
                "can_continue_with_sample": True,
            }

    extracted = extract_receipt_from_text(
        ocr_text,
        source_mode=extraction_mode,
    )
    extracted["provenance"] = provenance
    validation = validate_receipt(extracted)
    if review_record is None:
        review_record = {
            "decision": "APPROVED" if human_approved else "PENDING",
            "reviewer": "learner" if human_approved else "",
            "reviewed_at": "",
            "note": "",
        }
    decision = review_record.get("decision", "PENDING")
    review_status = (
        decision
        if validation["valid"] and decision in {"APPROVED", "CHANGED"}
        else "PENDING_REVIEW"
        if validation["valid"]
        else "BLOCKED_BY_VALIDATION"
    )
    xlsx_bytes = (
        receipt_to_xlsx_bytes(
            extracted,
            source_text=ocr_text,
            review_status=review_status,
            review_record=review_record,
        )
        if review_status in {"APPROVED", "CHANGED"}
        else None
    )

    return {
        "ok": True,
        "status": mode,
        "ocr_text": ocr_text,
        "data": extracted,
        "validation": validation,
        "review_status": review_status,
        "review_record": review_record,
        "xlsx_bytes": xlsx_bytes,
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
