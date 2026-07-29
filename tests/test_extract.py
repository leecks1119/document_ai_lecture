import pytest

from src.extract import (
    build_extraction_prompt,
    extract_receipt_from_text,
    mock_extract,
    validate_schema,
)
from src.sample_data import (
    GOLDEN_RECEIPT_OCR_TEXT,
    SAMPLE_OCR_TEXT,
    SAMPLE_VLM_MARKDOWN,
)


def test_mock_extract_returns_expected_receipt():
    result = mock_extract(SAMPLE_OCR_TEXT)

    assert result["store_name"] == "샘플문구점"
    assert result["date"] == "2026-07-27"
    assert result["total_amount"] == 5000
    assert [item["name"] for item in result["items"]] == ["연필", "노트"]


def test_prompt_requires_null_for_missing_values():
    prompt = build_extraction_prompt(SAMPLE_OCR_TEXT)

    assert "null" in prompt
    assert SAMPLE_OCR_TEXT in prompt


def test_mock_extract_reads_paddleocr_vl_markdown_table():
    result = mock_extract(SAMPLE_VLM_MARKDOWN)

    assert result["store_name"] == "샘플문구점"
    assert result["total_amount"] == 5000
    assert [item["name"] for item in result["items"]] == ["연필", "노트"]


def test_sample_matches_schema_when_jsonschema_is_available():
    pytest.importorskip("jsonschema")

    assert validate_schema(mock_extract(SAMPLE_OCR_TEXT)) == []


def test_public_korean_receipt_variants_are_parsed():
    result = extract_receipt_from_text(
        GOLDEN_RECEIPT_OCR_TEXT,
        source_mode="course_example_rule_extraction",
    )

    assert result["store_name"] == "이태리집"
    assert result["date"] == "2025-10-04"
    assert result["total_amount"] == 76000
    assert len(result["items"]) == 5
    assert result["items"][-1] == {
        "name": "콜라",
        "quantity": 3,
        "unit_price": 2000,
        "line_total": 6000,
    }
    assert result["evidence"]["total_amount"]["raw_value"] == "합계 금액 76,000"


def test_actual_paddleocr_vl_html_receipt_table_is_parsed():
    actual_vlm_markdown = """# [영수증] 이태리집 / 강원특별자치도 태백시 민영로 262(황지동)
2025-10-04 12:33:37
<table>
<tr><td>상품명</td><td>단가</td><td>수량 금액</td></tr>
<tr><td>패퍼로디엔칩스</td><td>29,000</td><td>129,000</td></tr>
<tr><td>콜라</td><td>2,000</td><td>36,000</td></tr>
<tr><td>합계금액</td><td colspan="2">76,000</td></tr>
<tr><td>부가세 과세물품가액</td><td colspan="2">69,094</td></tr>
<tr><td>부가세</td><td colspan="2">6,906</td></tr>
<tr><td colspan="3">*** 현금영수증(소득공제)[1] ***</td></tr>
</table>"""

    result = extract_receipt_from_text(
        actual_vlm_markdown,
        source_mode="paddleocr_vl_1_6_actual_inference",
    )

    assert result["store_name"] == "이태리집"
    assert result["date"] == "2025-10-04"
    assert result["total_amount"] == 76000
    assert result["evidence"]["total_amount"]["raw_value"] == "합계금액\t76,000"
    assert result["items"] == [
        {
            "name": "패퍼로디엔칩스",
            "quantity": 1,
            "unit_price": 29000,
            "line_total": 29000,
        },
        {
            "name": "콜라",
            "quantity": 3,
            "unit_price": 2000,
            "line_total": 6000,
        },
    ]
    assert result["tax_breakdown"] == {
        "mode": "included_in_item_prices",
        "supply_amount": 69094,
        "vat": 6906,
        "payable_total": 76000,
    }


def test_total_uses_last_amount_when_ocr_adds_a_stray_token():
    result = extract_receipt_from_text("이태리집\n합계 9 76,000")

    assert result["total_amount"] == 76000
    assert result["evidence"]["total_amount"]["raw_value"] == "합계 9 76,000"


def test_unknown_values_remain_none():
    result = extract_receipt_from_text("상호만있는문서")

    assert result["date"] is None
    assert result["total_amount"] is None
    assert result["items"] == []
