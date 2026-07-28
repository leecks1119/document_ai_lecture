"""2~8교시와 데모 앱에서 사용하는 골든 영수증·복구 데이터."""

from __future__ import annotations

from copy import deepcopy


SAMPLE_OCR_TEXT = """샘플문구점
거래일자: 2026-07-27
연필 2개 × 1,000원 = 2,000원
노트 1개 × 3,000원 = 3,000원
합계: 5,000원
"""

GOLDEN_RECEIPT_OCR_TEXT = """이태리집
거래일시 2025-10-04 12:33:37
페퍼로니 앤 치즈 29,000 1 29,000
토마토 파스타 14,000 1 14,000
수제 돈가스 13,000 1 13,000
새우 칠리치 필라 14,000 1 14,000
콜라 2,000 3 6,000
합계 금액 76,000
부가세 과세물품가액 69,094
부가세 6,906
"""

GOLDEN_RECEIPT_VLM_MARKDOWN = """# 이태리집

> **수업용 VLM 구조 예제** — 현재 실행에서 VLM을 호출한 결과가 아닙니다.

거래일시: 2025-10-04 12:33:37

| 품목 | 수량 | 단가 | 금액 |
| --- | ---: | ---: | ---: |
| 페퍼로니 앤 치즈 | 1 | 29,000원 | 29,000원 |
| 토마토 파스타 | 1 | 14,000원 | 14,000원 |
| 수제 돈가스 | 1 | 13,000원 | 13,000원 |
| 새우 칠리치 필라 | 1 | 14,000원 | 14,000원 |
| 콜라 | 3 | 2,000원 | 6,000원 |

**합계: 76,000원**

부가세 과세물품가액 69,094
부가세 6,906
"""

GOLDEN_RECEIPT_VLM_RESULT = {
    "target_technology": "PaddleOCR-VL-1.6",
    "executed_model": None,
    "source_mode": "course_example_vlm_structure",
    "provenance": {
        "fixture_type": "course_example",
        "input_file": "taebaek_restaurant_2025_redacted.png",
        "engine": "not_executed",
        "engine_version": "not_applicable",
        "target_technology": "PaddleOCR-VL-1.6",
        "created_by": "course maintainer",
        "disclaimer": "현재 실행에서 VLM을 호출한 결과가 아닙니다.",
    },
    "pages": [
        {
            "page": 1,
            "markdown": GOLDEN_RECEIPT_VLM_MARKDOWN,
            "blocks": [
                {"label": "title", "content": "이태리집", "order": 1},
                {
                    "label": "text",
                    "content": "거래일시: 2025-10-04 12:33:37",
                    "order": 2,
                },
                {
                    "label": "table",
                    "content": "품목·수량·단가·금액 5행",
                    "order": 3,
                },
                {"label": "text", "content": "합계: 76,000원", "order": 4},
            ],
        }
    ],
}

SAMPLE_OCR_RESULT = [
    {
        "box": [[80, 70], [330, 70], [330, 125], [80, 125]],
        "text": "샘플문구점",
        "confidence": 0.98,
    },
    {
        "box": [[80, 160], [500, 160], [500, 205], [80, 205]],
        "text": "거래일자: 2026-07-27",
        "confidence": 0.97,
    },
    {
        "box": [[80, 270], [650, 270], [650, 315], [80, 315]],
        "text": "연필 2개 × 1,000원 = 2,000원",
        "confidence": 0.93,
    },
    {
        "box": [[80, 340], [650, 340], [650, 385], [80, 385]],
        "text": "노트 1개 × 3,000원 = 3,000원",
        "confidence": 0.95,
    },
    {
        "box": [[80, 465], [440, 465], [440, 520], [80, 520]],
        "text": "합계: 5,000원",
        "confidence": 0.99,
    },
]

SAMPLE_VLM_MARKDOWN = """# 샘플문구점

> **수업용 합성 예제** — 현재 실행에서 VLM을 호출한 결과가 아닙니다.

거래일자: 2026-07-27

| 품목 | 수량 | 단가 | 금액 |
| --- | ---: | ---: | ---: |
| 연필 | 2 | 1,000원 | 2,000원 |
| 노트 | 1 | 3,000원 | 3,000원 |

**합계: 5,000원**
"""

SAMPLE_VLM_RESULT = {
    "target_technology": "PaddleOCR-VL-1.6",
    "executed_model": None,
    "source_mode": "synthetic_fixture",
    "provenance": {
        "fixture_type": "synthetic_fixture",
        "input_file": "receipt_sample.png",
        "created_by": "course generator",
        "disclaimer": "현재 실행에서 VLM을 호출한 결과가 아닙니다.",
    },
    "pages": [
        {
            "page": 1,
            "markdown": SAMPLE_VLM_MARKDOWN,
            "blocks": [
                {"label": "title", "content": "샘플문구점", "order": 1},
                {"label": "text", "content": "거래일자: 2026-07-27", "order": 2},
                {
                    "label": "table",
                    "content": "| 품목 | 수량 | 단가 | 금액 |",
                    "order": 3,
                },
                {"label": "text", "content": "합계: 5,000원", "order": 4},
            ],
        }
    ],
}

SAMPLE_RECEIPT = {
    "document_type": "receipt",
    "store_name": "샘플문구점",
    "date": "2026-07-27",
    "total_amount": 5000,
    "items": [
        {"name": "연필", "quantity": 2, "unit_price": 1000, "line_total": 2000},
        {"name": "노트", "quantity": 1, "unit_price": 3000, "line_total": 3000},
    ],
    "adjustments": {
        "discount": 0,
        "tax": 0,
        "service": 0,
        "rounding": 0,
    },
    "evidence": {
        "store_name": {"raw_value": "샘플문구점", "line": 1},
        "date": {"raw_value": "거래일자: 2026-07-27", "line": 2},
        "total_amount": {"raw_value": "합계: 5,000원", "line": 5},
    },
    "raw_values": {
        "store_name": "샘플문구점",
        "date": "2026-07-27",
        "total_amount": "5,000원",
    },
    "cleaned_values": {
        "store_name": "샘플문구점",
        "date": "2026-07-27",
        "total_amount": 5000,
    },
    "source_mode": "synthetic_fixture_rule_extraction",
}

GOLDEN_RECEIPT = {
    "document_type": "receipt",
    "store_name": "이태리집",
    "date": "2025-10-04",
    "total_amount": 76000,
    "items": [
        {
            "name": "페퍼로니 앤 치즈",
            "quantity": 1,
            "unit_price": 29000,
            "line_total": 29000,
        },
        {
            "name": "토마토 파스타",
            "quantity": 1,
            "unit_price": 14000,
            "line_total": 14000,
        },
        {
            "name": "수제 돈가스",
            "quantity": 1,
            "unit_price": 13000,
            "line_total": 13000,
        },
        {
            "name": "새우 칠리치 필라",
            "quantity": 1,
            "unit_price": 14000,
            "line_total": 14000,
        },
        {
            "name": "콜라",
            "quantity": 3,
            "unit_price": 2000,
            "line_total": 6000,
        },
    ],
    "adjustments": {
        "discount": 0,
        "tax": 0,
        "service": 0,
        "rounding": 0,
    },
    "tax_breakdown": {
        "mode": "included_in_item_prices",
        "supply_amount": 69094,
        "vat": 6906,
        "payable_total": 76000,
    },
    "evidence": {
        "store_name": {"raw_value": "이태리집", "line": 1},
        "date": {
            "raw_value": "거래일시 2025-10-04 12:33:37",
            "line": 2,
        },
        "total_amount": {"raw_value": "합계 금액 76,000", "line": 8},
    },
    "raw_values": {
        "store_name": "이태리집",
        "date": "2025-10-04 12:33:37",
        "total_amount": "76,000",
    },
    "cleaned_values": {
        "store_name": "이태리집",
        "date": "2025-10-04",
        "total_amount": 76000,
    },
    "source_mode": "course_example_rule_extraction",
}

MISSING_STORE_RECEIPT = {
    **SAMPLE_RECEIPT,
    "store_name": None,
    "items": deepcopy(SAMPLE_RECEIPT["items"]),
}

WRONG_TOTAL_RECEIPT = {
    **SAMPLE_RECEIPT,
    "total_amount": 6000,
    "items": deepcopy(SAMPLE_RECEIPT["items"]),
}


def sample_receipt() -> dict:
    """호출한 코드가 원본 상수를 바꾸지 않도록 복사본을 반환한다."""

    return deepcopy(SAMPLE_RECEIPT)
