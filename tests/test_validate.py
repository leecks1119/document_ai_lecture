from copy import deepcopy

from src.sample_data import GOLDEN_RECEIPT, SAMPLE_RECEIPT
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
    assert any("품목·조정 후 합계" in error for error in result["errors"])


def test_item_math_and_boolean_amount_are_blocked():
    data = deepcopy(SAMPLE_RECEIPT)
    data["items"][0]["line_total"] = True

    result = validate_receipt(data)

    assert not result["valid"]
    assert any("0 이상의 정수" in error for error in result["errors"])


def test_missing_evidence_is_warning_not_silent():
    data = deepcopy(SAMPLE_RECEIPT)
    data["evidence"].pop("total_amount")

    result = validate_receipt(data)

    assert result["valid"]
    assert "total_amount의 원본 근거가 없습니다." in result["warnings"]


def test_adjustments_are_included_in_reconciliation():
    data = deepcopy(SAMPLE_RECEIPT)
    data["adjustments"]["discount"] = 500
    data["total_amount"] = 4500

    assert validate_receipt(data)["valid"]


def test_included_vat_is_not_added_twice():
    data = deepcopy(GOLDEN_RECEIPT)

    assert validate_receipt(data)["valid"]

    data["adjustments"]["tax"] = data["tax_breakdown"]["vat"]
    result = validate_receipt(data)
    assert not result["valid"]
    assert any("이중 계산" in error for error in result["errors"])
