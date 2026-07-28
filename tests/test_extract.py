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
        source_mode="prepared_fixture_rule_extraction",
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


def test_total_uses_last_amount_when_ocr_adds_a_stray_token():
    result = extract_receipt_from_text("이태리집\n합계 9 76,000")

    assert result["total_amount"] == 76000
    assert result["evidence"]["total_amount"]["raw_value"] == "합계 9 76,000"


def test_unknown_values_remain_none():
    result = extract_receipt_from_text("상호만있는문서")

    assert result["date"] is None
    assert result["total_amount"] is None
    assert result["items"] == []
