"""3교시: OCR 원문을 잃지 않고 정리하는 함수."""

from __future__ import annotations

import re


def normalize_line(line: str) -> tuple[str, list[str]]:
    """한 줄의 불필요한 공백만 정리하고 변경 기록을 반환한다."""

    original = line
    cleaned = line.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    changes: list[str] = []
    if original != cleaned:
        changes.append(f"공백 정리: {original!r} → {cleaned!r}")
    return cleaned, changes


def group_receipt_lines(raw_text: str) -> dict:
    """영수증 OCR 텍스트를 헤더·날짜·품목·합계 줄로 분류한다."""

    cleaned_lines: list[str] = []
    change_log: list[str] = []

    for raw_line in raw_text.splitlines():
        cleaned, changes = normalize_line(raw_line)
        if cleaned:
            cleaned_lines.append(cleaned)
            change_log.extend(changes)

    groups = {
        "header": [],
        "date": [],
        "items": [],
        "total": [],
        "other": [],
    }

    for line in cleaned_lines:
        if "거래일자" in line:
            groups["date"].append(line)
        elif "합계" in line:
            groups["total"].append(line)
        elif "개" in line and ("×" in line or "x" in line.lower()):
            groups["items"].append(line)
        elif not groups["header"]:
            groups["header"].append(line)
        else:
            groups["other"].append(line)

    return {
        "raw_text": raw_text,
        "cleaned_lines": cleaned_lines,
        "groups": groups,
        "change_log": change_log,
    }
