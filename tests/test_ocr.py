import json
import fitz
import pytest
from pathlib import Path

from src.extract import extract_receipt_from_text
from src.ocr import (
    _prepare_image_paths,
    ocr_text_from_result,
    reconstruct_spatial_lines,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "ppocrv5_live_receipt_tokens.json"
)


def test_recorded_ppocrv5_tokens_restore_receipt_rows():
    tokens = json.loads(FIXTURE.read_text(encoding="utf-8"))

    lines = reconstruct_spatial_lines(tokens)
    text = ocr_text_from_result(tokens)
    receipt = extract_receipt_from_text(
        text,
        source_mode="recorded_ppocrv5_regression",
    )

    assert "페퍼르니인집스 29,000 1 29,000" in lines
    assert "합계 9 76,000" in lines
    assert receipt["date"] == "2025-10-04"
    assert receipt["total_amount"] == 76000
    assert len(receipt["items"]) == 5
    assert receipt["items"][-1]["line_total"] == 6000


def test_unpositioned_fixture_keeps_input_order():
    result = [
        {"text": "첫 줄", "box": []},
        {"text": "둘째 줄", "box": None},
    ]

    assert reconstruct_spatial_lines(result) == ["첫 줄", "둘째 줄"]


def test_corrupt_pdf_has_beginner_friendly_error(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(ValueError, match="PDF 파일을 열 수 없습니다"):
        _prepare_image_paths(path, tmp_path / "output")


def test_encrypted_pdf_is_rejected(tmp_path):
    path = tmp_path / "encrypted.pdf"
    document = fitz.open()
    document.new_page()
    document.save(
        path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="learner",
    )
    document.close()

    with pytest.raises(ValueError, match="암호가 설정된 PDF"):
        _prepare_image_paths(path, tmp_path / "output")


def test_pdf_over_three_pages_is_rejected(tmp_path):
    path = tmp_path / "four-pages.pdf"
    document = fitz.open()
    for _ in range(4):
        document.new_page()
    document.save(path)
    document.close()

    with pytest.raises(ValueError, match="최대 3페이지"):
        _prepare_image_paths(path, tmp_path / "output")
