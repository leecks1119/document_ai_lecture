"""8교시에서 비교하는 견적서·신청서·거래명세서 예시."""

from __future__ import annotations

from copy import deepcopy


DOCUMENT_EXAMPLES = {
    "quotation": {
        "document_type": "quotation",
        "document_number": "Q-2026-0728-01",
        "supplier": "한빛오피스",
        "customer": "새봄기획",
        "quote_date": "2026-07-28",
        "valid_until": "2026-08-11",
        "items": [
            {
                "name": "회의용 의자",
                "quantity": 4,
                "unit_price": 85000,
                "line_total": 340000,
            },
            {
                "name": "이동식 화이트보드",
                "quantity": 1,
                "unit_price": 160000,
                "line_total": 160000,
            },
        ],
        "subtotal": 500000,
        "tax": 50000,
        "total_amount": 550000,
    },
    "application": {
        "document_type": "application",
        "application_id": "EDU-2026-014",
        "applicant_name": "김하늘",
        "department": "업무혁신팀",
        "requested_program": "Document AI 업무 자동화",
        "requested_date": "2026-08-18",
        "manager_approval": "승인",
        "privacy_consent": True,
    },
    "transaction_statement": {
        "document_type": "transaction_statement",
        "statement_number": "TS-2026-0728-03",
        "supplier": "다온유통",
        "customer": "푸른상사",
        "transaction_date": "2026-07-28",
        "items": [
            {
                "name": "A4 복사용지",
                "quantity": 10,
                "unit_price": 6500,
                "line_total": 65000,
            },
            {
                "name": "검정 볼펜",
                "quantity": 20,
                "unit_price": 900,
                "line_total": 18000,
            },
        ],
        "subtotal": 83000,
        "tax": 8300,
        "total_amount": 91300,
    },
}


REQUIRED_FIELDS = {
    "quotation": (
        "document_number",
        "supplier",
        "customer",
        "quote_date",
        "items",
        "total_amount",
    ),
    "application": (
        "application_id",
        "applicant_name",
        "department",
        "requested_program",
        "requested_date",
        "manager_approval",
    ),
    "transaction_statement": (
        "statement_number",
        "supplier",
        "customer",
        "transaction_date",
        "items",
        "total_amount",
    ),
}


def document_example(document_type: str) -> dict:
    """수업 코드가 상수를 바꾸지 않도록 복사본을 반환한다."""

    return deepcopy(DOCUMENT_EXAMPLES[document_type])


def validate_document_example(data: dict) -> dict:
    """8교시 PoC 검토에 필요한 최소 공통 검증을 수행한다."""

    document_type = data.get("document_type")
    errors: list[str] = []
    warnings: list[str] = []
    if document_type not in REQUIRED_FIELDS:
        return {
            "valid": False,
            "warnings": [],
            "errors": ["지원 문서 유형이 아닙니다."],
        }

    for field in REQUIRED_FIELDS[document_type]:
        if data.get(field) in (None, "", []):
            errors.append(f"필수값 누락: {field}")

    if document_type in {"quotation", "transaction_statement"}:
        items = data.get("items") or []
        subtotal = 0
        for index, item in enumerate(items, start=1):
            quantity = item.get("quantity")
            unit_price = item.get("unit_price")
            line_total = item.get("line_total")
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (quantity, unit_price, line_total)
            ):
                errors.append(f"{index}번째 품목 금액 형식을 확인하세요.")
                continue
            if quantity * unit_price != line_total:
                errors.append(f"{index}번째 품목 수량×단가가 맞지 않습니다.")
            subtotal += line_total
        if subtotal != data.get("subtotal"):
            errors.append("품목 소계와 subtotal이 다릅니다.")
        if subtotal + data.get("tax", 0) != data.get("total_amount"):
            errors.append("소계+세액과 총액이 다릅니다.")

    if document_type == "application" and not data.get("privacy_consent"):
        warnings.append("개인정보 동의 여부를 사람이 확인해야 합니다.")

    return {"valid": not errors, "warnings": warnings, "errors": errors}
