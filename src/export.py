"""7교시: 영수증 JSON을 안전한 CSV로 바꾼다."""

from __future__ import annotations

import csv
import io
from typing import Any


FORMULA_PREFIXES = ("=", "+", "-", "@")


def safe_spreadsheet_text(value: Any) -> Any:
    """스프레드시트에서 수식으로 해석될 수 있는 문자열을 보호한다."""

    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def receipt_to_rows(data: dict) -> list[dict]:
    """영수증의 반복 품목을 CSV의 여러 행으로 펼친다."""

    base = {
        "store_name": data.get("store_name"),
        "date": data.get("date"),
        "total_amount": data.get("total_amount"),
    }
    rows = []
    for item in data.get("items") or []:
        row = {
            **base,
            "item_name": item.get("name"),
            "quantity": item.get("quantity"),
            "unit_price": item.get("unit_price"),
            "line_total": item.get("line_total"),
        }
        rows.append(
            {key: safe_spreadsheet_text(value) for key, value in row.items()}
        )
    return rows


def receipt_to_csv_bytes(data: dict) -> bytes:
    """Excel에서도 한글을 읽기 쉬운 UTF-8 BOM CSV 바이트를 반환한다."""

    rows = receipt_to_rows(data)
    columns = [
        "store_name",
        "date",
        "total_amount",
        "item_name",
        "quantity",
        "unit_price",
        "line_total",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")
