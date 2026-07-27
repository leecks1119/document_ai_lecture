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

    if data.get("date") and not _is_iso_date(data["date"]):
        errors.append("date는 YYYY-MM-DD 형식이어야 합니다.")

    total_amount = data.get("total_amount")
    if total_amount is not None and (
        not isinstance(total_amount, int) or total_amount < 0
    ):
        errors.append("total_amount는 0 이상의 정수여야 합니다.")

    items = data.get("items") or []
    if items:
        line_total_sum = sum(
            item.get("line_total", 0)
            for item in items
            if isinstance(item.get("line_total"), int)
        )
        if isinstance(total_amount, int) and line_total_sum != total_amount:
            errors.append(
                f"품목 합계 {line_total_sum:,}원과 총액 {total_amount:,}원이 다릅니다."
            )

        for index, item in enumerate(items, start=1):
            if not item.get("name"):
                warnings.append(f"{index}번째 품목 이름을 확인하세요.")

    return {
        "valid": not errors,
        "warnings": warnings,
        "errors": errors,
    }
