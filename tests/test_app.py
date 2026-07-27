import pytest


gradio = pytest.importorskip("gradio")

from app import build_demo, run_sample


def test_gradio_demo_builds_without_launching():
    assert isinstance(build_demo(), gradio.Blocks)


def test_sample_handler_labels_mock_output():
    status, ocr_text, data, rows, csv_path = run_sample()

    assert "MOCK OCR + MOCK 추출" in status
    assert "샘플문구점" in ocr_text
    assert data["total_amount"] == 5000
    assert len(rows) == 2
    assert csv_path
