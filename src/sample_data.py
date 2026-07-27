"""2~8교시와 데모 앱에서 사용하는 합성 영수증 데이터."""

from __future__ import annotations

from copy import deepcopy


SAMPLE_OCR_TEXT = """샘플문구점
거래일자: 2026-07-27
연필 2개 × 1,000원 = 2,000원
노트 1개 × 3,000원 = 3,000원
합계: 5,000원
"""

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

거래일자: 2026-07-27

| 품목 | 수량 | 단가 | 금액 |
| --- | ---: | ---: | ---: |
| 연필 | 2 | 1,000원 | 2,000원 |
| 노트 | 1 | 3,000원 | 3,000원 |

**합계: 5,000원**
"""

SAMPLE_VLM_RESULT = {
    "model": "PaddleOCR-VL-1.6",
    "source_mode": "mock_vlm",
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
    "source_mode": "mock",
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
