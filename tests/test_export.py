import io

from openpyxl import load_workbook

from src.export import receipt_to_rows, receipt_to_xlsx_bytes, safe_spreadsheet_text
from src.sample_data import SAMPLE_RECEIPT


def test_receipt_items_become_excel_rows():
    rows = receipt_to_rows(SAMPLE_RECEIPT)

    assert len(rows) == 2
    assert rows[0]["item_name"] == "연필"


def test_xlsx_has_expected_sheets_and_values():
    xlsx_bytes = receipt_to_xlsx_bytes(
        SAMPLE_RECEIPT,
        source_text="샘플 OCR 원문",
    )
    workbook = load_workbook(io.BytesIO(xlsx_bytes))

    assert workbook.sheetnames == ["검토_요약", "품목", "원문"]
    assert workbook["품목"]["A2"].value == "샘플문구점"
    assert workbook["원문"]["B2"].value == "샘플 OCR 원문"


def test_formula_prefix_is_escaped():
    assert safe_spreadsheet_text("=1+1") == "'=1+1"


def test_formula_prefix_is_escaped_in_xlsx():
    data = {
        **SAMPLE_RECEIPT,
        "store_name": "=HYPERLINK(\"https://example.com\")",
    }
    workbook = load_workbook(io.BytesIO(receipt_to_xlsx_bytes(data)))

    assert workbook["검토_요약"]["D2"].value.startswith("'=")
