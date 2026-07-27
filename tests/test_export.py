from src.export import receipt_to_csv_bytes, receipt_to_rows, safe_spreadsheet_text
from src.sample_data import SAMPLE_RECEIPT


def test_receipt_items_become_csv_rows():
    rows = receipt_to_rows(SAMPLE_RECEIPT)

    assert len(rows) == 2
    assert rows[0]["item_name"] == "연필"


def test_csv_has_utf8_bom():
    csv_bytes = receipt_to_csv_bytes(SAMPLE_RECEIPT)

    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    assert "샘플문구점" in csv_bytes.decode("utf-8-sig")


def test_formula_prefix_is_escaped():
    assert safe_spreadsheet_text("=1+1") == "'=1+1"
