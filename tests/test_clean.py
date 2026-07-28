from src.clean import group_receipt_lines, normalize_line
from src.sample_data import SAMPLE_OCR_TEXT


def test_normalize_line_records_changes():
    cleaned, changes = normalize_line("  합계:   5,000원  ")

    assert cleaned == "합계: 5,000원"
    assert changes


def test_group_receipt_lines_preserves_raw_text():
    result = group_receipt_lines(SAMPLE_OCR_TEXT)

    assert result["raw_text"] == SAMPLE_OCR_TEXT
    assert len(result["groups"]["items"]) == 2
    assert result["groups"]["total"] == ["합계: 5,000원"]
