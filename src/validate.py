"""7교시: 영수증 결과의 필수값과 합계를 확인한다."""

from __future__ import annotations

from datetime import date
from typing import Any


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def validate_receipt(data: dict) -> dict:
    """초보자 수업에 필요한 최소 영수증 규칙만 검사한다."""

    warnings: list[str] = []
    errors: list[str] = []

    for field in ("store_name", "date", "total_amount", "items"):
        if data.get(field) in (None, "", []):
            errors.append(f"필수값 누락: {field}")

    receipt_date = data.get("date")
    if receipt_date and not _is_iso_date(receipt_date):
        errors.append("date는 YYYY-MM-DD 형식이어야 합니다.")
    elif receipt_date and date.fromisoformat(receipt_date) > date.today():
        warnings.append("date가 오늘보다 미래입니다. 원본 날짜를 확인하세요.")

    total_amount = data.get("total_amount")
    if total_amount is not None and (
        isinstance(total_amount, bool)
        or not isinstance(total_amount, int)
        or total_amount < 0
    ):
        errors.append("total_amount는 0 이상의 정수여야 합니다.")

    items = data.get("items") or []
    if items:
        line_total_sum = 0
        for index, item in enumerate(items, start=1):
            if not item.get("name"):
                errors.append(f"{index}번째 품목 이름이 비어 있습니다.")
            for field in ("quantity", "unit_price", "line_total"):
                value = item.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    errors.append(
                        f"{index}번째 품목의 {field}는 0 이상의 정수여야 합니다."
                    )
            quantity = item.get("quantity")
            unit_price = item.get("unit_price")
            line_total = item.get("line_total")
            if (
                isinstance(quantity, int)
                and not isinstance(quantity, bool)
                and isinstance(unit_price, int)
                and not isinstance(unit_price, bool)
                and isinstance(line_total, int)
                and not isinstance(line_total, bool)
            ):
                line_total_sum += line_total
                if quantity * unit_price != line_total:
                    errors.append(
                        f"{index}번째 품목: 수량×단가와 품목 금액이 다릅니다."
                    )

        adjustments = data.get("adjustments") or {}
        adjustment_total = sum(
            value
            for value in (
                -adjustments.get("discount", 0),
                adjustments.get("tax", 0),
                adjustments.get("service", 0),
                adjustments.get("rounding", 0),
            )
            if isinstance(value, int) and not isinstance(value, bool)
        )
        expected_total = line_total_sum + adjustment_total
        if (
            isinstance(total_amount, int)
            and not isinstance(total_amount, bool)
            and expected_total != total_amount
        ):
            errors.append(
                f"품목·조정 후 합계 {expected_total:,}원과 총액 "
                f"{total_amount:,}원이 다릅니다."
            )

    tax_breakdown = data.get("tax_breakdown")
    if tax_breakdown and tax_breakdown.get("mode") == "included_in_item_prices":
        supply_amount = tax_breakdown.get("supply_amount")
        vat = tax_breakdown.get("vat")
        payable_total = tax_breakdown.get("payable_total")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (supply_amount, vat, payable_total)
        ):
            errors.append("포함세액 내역은 정수 금액이어야 합니다.")
        elif supply_amount + vat != payable_total:
            errors.append("공급가액과 포함 부가세의 합이 결제금액과 다릅니다.")
        elif payable_total != total_amount:
            errors.append("포함세액 내역의 결제금액과 영수증 총액이 다릅니다.")
        if (data.get("adjustments") or {}).get("tax", 0) != 0:
            errors.append(
                "포함 부가세를 adjustments.tax에 다시 더하면 이중 계산됩니다."
            )

    evidence = data.get("evidence") or {}
    for field in ("store_name", "date", "total_amount"):
        if data.get(field) not in (None, "") and not evidence.get(field):
            warnings.append(f"{field}의 원본 근거가 없습니다.")

    return {
        "valid": not errors,
        "warnings": warnings,
        "errors": errors,
    }
