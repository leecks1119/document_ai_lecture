import json
import hashlib
import fitz
import pytest
from pathlib import Path
from PIL import Image

from src.extract import extract_receipt_from_text
from src.ocr import (
    _prepare_image_paths,
    ocr_text_from_result,
    reconstruct_spatial_lines,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "ppocrv5_recorded_receipt_tokens.json"
)
METADATA_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "ppocrv5_recorded_receipt_metadata.json"
)
ROOT = Path(__file__).resolve().parents[1]


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


def test_recorded_boxes_use_the_same_aspect_ratio_as_the_source_image():
    tokens = json.loads(FIXTURE.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_FIXTURE.read_text(encoding="utf-8"))
    source_path = ROOT / metadata["source_image"]
    source_bytes = source_path.read_bytes()

    assert hashlib.sha256(source_bytes).hexdigest() == metadata["source_image_sha256"]
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == metadata["token_file_sha256"]
    with Image.open(source_path) as image:
        source_size = image.size

    assert source_size == (
        metadata["source_image_size"]["width"],
        metadata["source_image_size"]["height"],
    )
    coordinate_size = (
        metadata["coordinate_space"]["width"],
        metadata["coordinate_space"]["height"],
    )
    assert coordinate_size[1] == round(
        source_size[1] * coordinate_size[0] / source_size[0]
    )
    assert metadata["token_count"] == len(tokens)

    max_x = max(point[0] for token in tokens for point in token["box"])
    max_y = max(point[1] for token in tokens for point in token["box"])
    assert max_x <= coordinate_size[0]
    assert max_y <= coordinate_size[1]

    scale_x = source_size[0] / coordinate_size[0]
    scale_y = source_size[1] / coordinate_size[1]
    assert abs(scale_x - scale_y) < 0.01


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
