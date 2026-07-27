import fitz
import pytest

from src.ocr import _prepare_image_paths


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
