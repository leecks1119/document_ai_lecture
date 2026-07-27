from copy import deepcopy

from src.sample_data import SAMPLE_RECEIPT
from src.validate import validate_receipt


def test_valid_receipt_passes():
    assert validate_receipt(deepcopy(SAMPLE_RECEIPT)) == {
        "valid": True,
        "warnings": [],
        "errors": [],
    }


def test_missing_required_field_fails():
    data = deepcopy(SAMPLE_RECEIPT)
    data["store_name"] = None

    result = validate_receipt(data)

    assert not result["valid"]
    assert "필수값 누락: store_name" in result["errors"]


def test_wrong_item_total_fails():
    data = deepcopy(SAMPLE_RECEIPT)
    data["total_amount"] = 6000

    result = validate_receipt(data)

    assert not result["valid"]
    assert any("품목 합계" in error for error in result["errors"])
