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


def mock_extract(ocr_text: str) -> dict:
    """수업용 합성 영수증을 규칙으로 구조화한다.

    이 함수는 생성형 AI 호출 결과의 *형태*를 연습하기 위한 mock이다.
    실제 모델 호출이나 일반적인 영수증 인식 성능을 흉내 내지 않는다.
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
                "source_mode": "mock",
            }
        )
        return empty

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", ocr_text)
    total_match = re.search(r"합계\s*:\s*([\d,]+)원", ocr_text)
    item_pattern = re.compile(
        r"(?P<name>.+?)\s+(?P<quantity>\d+)개\s*[×x]\s*"
        r"(?P<unit>[\d,]+)원\s*=\s*(?P<line>[\d,]+)원"
    )
    markdown_item_pattern = re.compile(
        r"^\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<quantity>\d+)\s*\|"
        r"\s*(?P<unit>[\d,]+)원\s*\|\s*(?P<line>[\d,]+)원\s*\|$"
    )

    items = []
    for line in lines:
        match = item_pattern.search(line)
        if not match:
            match = markdown_item_pattern.search(line)
        if match:
            items.append(
                {
                    "name": match.group("name").strip(),
                    "quantity": int(match.group("quantity")),
                    "unit_price": _to_int(match.group("unit")),
                    "line_total": _to_int(match.group("line")),
                }
            )

    return {
        "document_type": "receipt",
        "store_name": lines[0].lstrip("# ").strip() if lines else None,
        "date": date_match.group(0) if date_match else None,
        "total_amount": _to_int(total_match.group(1)) if total_match else None,
        "items": items,
        "source_mode": "mock",
    }


def validate_schema(data: dict) -> list[str]:
    """jsonschema가 있으면 스키마 오류를 쉬운 문장으로 반환한다."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return ["jsonschema가 없어 스키마 검사를 건너뛰었습니다."]

    validator = Draft202012Validator(RECEIPT_SCHEMA)
    return [error.message for error in validator.iter_errors(data)]
