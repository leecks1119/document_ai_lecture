from src.sample_data import SAMPLE_VLM_MARKDOWN
from src.vlm import load_mock_vlm, vlm_text_from_result


def test_mock_vlm_is_labeled_and_contains_layout_blocks():
    result = load_mock_vlm()

    assert result["model"] == "PaddleOCR-VL-1.6"
    assert result["source_mode"] == "mock_vlm"
    assert result["pages"][0]["blocks"][2]["label"] == "table"


def test_vlm_markdown_can_be_read_as_document_text():
    assert vlm_text_from_result(load_mock_vlm()) == SAMPLE_VLM_MARKDOWN
