import pytest

from src.extract import build_extraction_prompt, mock_extract, validate_schema
from src.sample_data import SAMPLE_OCR_TEXT


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


def test_sample_matches_schema_when_jsonschema_is_available():
    pytest.importorskip("jsonschema")

    assert validate_schema(mock_extract(SAMPLE_OCR_TEXT)) == []
