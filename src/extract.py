"""4교시: OCR 텍스트를 영수증 JSON으로 구조화하는 함수."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .sample_data import SAMPLE_RECEIPT


RECEIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "document_type",
        "store_name",
        "date",
        "total_amount",
        "items",
    ],
    "properties": {
        "document_type": {"const": "receipt"},
        "store_name": {"type": ["string", "null"]},
        "date": {
            "type": ["string", "null"],
            "description": "YYYY-MM-DD",
        },
        "total_amount": {"type": ["integer", "null"], "minimum": 0},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "quantity", "unit_price", "line_total"],
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "quantity": {"type": ["integer", "null"], "minimum": 0},
                    "unit_price": {"type": ["integer", "null"], "minimum": 0},
                    "line_total": {"type": ["integer", "null"], "minimum": 0},
                },
            },
        },
        "adjustments": {
            "type": "object",
            "properties": {
                "discount": {"type": "integer"},
                "tax": {"type": "integer"},
                "service": {"type": "integer"},
                "rounding": {"type": "integer"},
            },
        },
        "tax_breakdown": {
            "type": ["object", "null"],
            "description": "품목 가격에 이미 포함된 공급가액·부가세 표시",
        },
        "evidence": {"type": "object"},
        "raw_values": {"type": "object"},
        "cleaned_values": {"type": "object"},
        "provenance": {"type": "object"},
        "source_mode": {"type": "string"},
    },
}


def build_extraction_prompt(ocr_text: str) -> str:
    """생성형 AI에 전달할 수 있는 짧고 명확한 추출 프롬프트를 만든다."""

    schema_text = json.dumps(RECEIPT_SCHEMA, ensure_ascii=False, indent=2)
    return f"""역할:
당신은 영수증 정보 추출 도우미입니다.

목표:
OCR 텍스트에서 상호명, 날짜, 품목, 합계를 JSON으로 추출하세요.

제약조건:
- 원문에 없는 값은 추측하지 말고 null로 반환하세요.
- 금액은 쉼표와 '원'을 제외한 정수로 반환하세요.
- JSON 외의 설명은 반환하지 마세요.

JSON Schema:
{schema_text}

OCR 텍스트:
{ocr_text}
"""


def _to_int(value: str) -> int:
    return int(value.replace(",", ""))


def _normalize_date(value: str) -> str:
    parts = re.split(r"[-./]", value)
    return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def _find_store_name(lines: list[str]) -> tuple[str | None, int | None]:
    ignored = re.compile(
        r"^(?:\[?영수(?:증)?\]?|거래|사업자|대표|주소|전화|상품명|품명|상\s*품)",
        re.IGNORECASE,
    )
    for index, line in enumerate(lines, start=1):
        candidate = line.lstrip("# ").strip()
        if candidate and not ignored.search(candidate):
            return candidate, index
    return None, None


def extract_receipt_from_text(
    ocr_text: str,
    *,
    source_mode: str = "rule_extraction",
) -> dict:
    """초보자 실습용 최소 규칙으로 영수증 텍스트를 구조화한다.

    이 함수는 범용 영수증 AI가 아니다. 한국 영수증에서 자주 보이는 날짜,
    합계, `품명 단가 수량 금액` 표기와 수업용 표기를 다루며, 읽지 못한 값은
    추측하지 않고 ``None``으로 남긴다.
    """

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if not lines:
        empty = deepcopy(SAMPLE_RECEIPT)
        empty.update(
            {
                "store_name": None,
                "date": None,
                "total_amount": None,
                "items": [],
                "source_mode": source_mode,
            }
        )
        return empty

    date_match = re.search(r"\b(\d{4}[-./]\d{1,2}[-./]\d{1,2})\b", ocr_text)
    total_line_text = next(
        (
            line
            for line in lines
            if re.search(
                r"(?:합\s*계|결제\s*금액|총\s*액)",
                line,
                re.IGNORECASE,
            )
        ),
        None,
    )
    total_candidates = (
        re.findall(r"(?<![\d,])\d[\d,]*(?![\d,])", total_line_text)
        if total_line_text
        else []
    )
    total_raw = total_candidates[-1] if total_candidates else None
    supply_match = re.search(
        r"(?:부가세\s*)?과세물품가액\s*[:：]?\s*(?P<amount>[\d,]+)",
        ocr_text,
    )
    vat_match = re.search(
        r"^부가세(?!\s*과세물품가액)\s*[:：]?\s*(?P<amount>[\d,]+)",
        ocr_text,
        re.MULTILINE,
    )
    item_pattern = re.compile(
        r"(?P<name>.+?)\s+(?P<quantity>\d+)개\s*[×x]\s*"
        r"(?P<unit>[\d,]+)원\s*=\s*(?P<line>[\d,]+)원"
    )
    markdown_item_pattern = re.compile(
        r"^\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<quantity>\d+)\s*\|"
        r"\s*(?P<unit>[\d,]+)원\s*\|\s*(?P<line>[\d,]+)원\s*\|$"
    )
    receipt_column_pattern = re.compile(
        r"^(?P<name>.+?)\s+(?P<unit>[\d,]+)\s+"
        r"(?P<quantity>\d+)\s+(?P<line>[\d,]+)\s*원?$"
    )

    items = []
    item_evidence = []
    for line_number, line in enumerate(lines, start=1):
        match = item_pattern.search(line)
        if not match:
            match = markdown_item_pattern.search(line)
        if not match:
            match = receipt_column_pattern.search(line)
        if match:
            item = {
                "name": match.group("name").strip(),
                "quantity": int(match.group("quantity")),
                "unit_price": _to_int(match.group("unit")),
                "line_total": _to_int(match.group("line")),
            }
            items.append(item)
            item_evidence.append(
                {
                    "raw_value": line,
                    "line": line_number,
                    "normalized_value": item,
                }
            )

    store_name, store_line = _find_store_name(lines)
    normalized_date = _normalize_date(date_match.group(1)) if date_match else None
    total_amount = _to_int(total_raw) if total_raw else None
    supply_amount = (
        _to_int(supply_match.group("amount")) if supply_match else None
    )
    vat_amount = _to_int(vat_match.group("amount")) if vat_match else None
    total_line = next(
        (
            index
            for index, line in enumerate(lines, start=1)
            if total_line_text and line == total_line_text
        ),
        None,
    )

    return {
        "document_type": "receipt",
        "store_name": store_name,
        "date": normalized_date,
        "total_amount": total_amount,
        "items": items,
        "adjustments": {
            "discount": 0,
            "tax": 0,
            "service": 0,
            "rounding": 0,
        },
        "tax_breakdown": {
            "mode": "included_in_item_prices",
            "supply_amount": supply_amount,
            "vat": vat_amount,
            "payable_total": total_amount,
        }
        if supply_amount is not None and vat_amount is not None
        else None,
        "evidence": {
            "store_name": {
                "raw_value": store_name,
                "line": store_line,
            },
            "date": {
                "raw_value": date_match.group(0) if date_match else None,
                "line": next(
                    (
                        index
                        for index, line in enumerate(lines, start=1)
                        if date_match and date_match.group(0) in line
                    ),
                    None,
                ),
            },
            "total_amount": {
                "raw_value": total_line_text,
                "line": total_line,
            },
            "items": item_evidence,
        },
        "raw_values": {
            "store_name": store_name,
            "date": date_match.group(0) if date_match else None,
            "total_amount": total_raw,
        },
        "cleaned_values": {
            "store_name": store_name,
            "date": normalized_date,
            "total_amount": total_amount,
        },
        "source_mode": source_mode,
    }


def mock_extract(ocr_text: str) -> dict:
    """이전 교재 코드와의 호환을 위한 명시적 합성 fixture 별칭."""

    return extract_receipt_from_text(
        ocr_text,
        source_mode="synthetic_fixture_rule_extraction",
    )


def validate_schema(data: dict) -> list[str]:
    """jsonschema가 있으면 스키마 오류를 쉬운 문장으로 반환한다."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return ["jsonschema가 없어 스키마 검사를 건너뛰었습니다."]

    validator = Draft202012Validator(RECEIPT_SCHEMA)
    return [error.message for error in validator.iter_errors(data)]
