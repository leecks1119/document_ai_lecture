"""합성 영수증 이미지와 출처가 표시된 수업용 준비 결과를 재생성한다."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export import receipt_to_xlsx_bytes
from src.sample_data import (
    GOLDEN_RECEIPT,
    GOLDEN_RECEIPT_OCR_TEXT,
    SAMPLE_OCR_RESULT,
    SAMPLE_OCR_TEXT,
    SAMPLE_RECEIPT,
    SAMPLE_VLM_MARKDOWN,
    SAMPLE_VLM_RESULT,
)


SAMPLE_DOCS = ROOT / "sample_docs"
SAMPLE_OUTPUTS = ROOT / "sample_outputs"


def find_korean_font() -> str:
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "한글 폰트를 찾을 수 없습니다. Noto Sans CJK 또는 나눔고딕을 설치하세요."
    )


def create_receipt_image() -> Image.Image:
    font_path = find_korean_font()
    image = Image.new("RGB", (900, 620), "#f6f1e7")
    draw = ImageDraw.Draw(image)

    title = ImageFont.truetype(font_path, 50)
    body = ImageFont.truetype(font_path, 34)
    total = ImageFont.truetype(font_path, 43)
    small = ImageFont.truetype(font_path, 25)

    draw.rounded_rectangle(
        (35, 25, 865, 590),
        radius=18,
        fill="#fffdf8",
        outline="#2f3a45",
        width=3,
    )
    draw.text((75, 65), "샘플문구점", font=title, fill="#17202a")
    draw.line((75, 135, 825, 135), fill="#9aa4ad", width=2)
    draw.text((75, 165), "거래일자: 2026-07-27", font=body, fill="#17202a")
    draw.text(
        (75, 275),
        "연필 2개 × 1,000원 = 2,000원",
        font=body,
        fill="#17202a",
    )
    draw.text(
        (75, 345),
        "노트 1개 × 3,000원 = 3,000원",
        font=body,
        fill="#17202a",
    )
    draw.line((75, 430, 825, 430), fill="#9aa4ad", width=2)
    draw.text((75, 465), "합계: 5,000원", font=total, fill="#b42318")
    draw.text(
        (75, 545),
        "교육용 합성 문서 · 실제 개인정보 없음",
        font=small,
        fill="#667085",
    )
    return image


def main() -> None:
    SAMPLE_DOCS.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUTPUTS.mkdir(parents=True, exist_ok=True)

    receipt = create_receipt_image()
    receipt.save(SAMPLE_DOCS / "receipt_sample.png", optimize=True)
    fixed_pdf_time = time.gmtime(0)
    receipt.save(
        SAMPLE_DOCS / "receipt_sample.pdf",
        "PDF",
        resolution=150,
        creationDate=fixed_pdf_time,
        modDate=fixed_pdf_time,
    )

    low_quality = receipt.rotate(
        4,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor="#d8d8d8",
    ).filter(ImageFilter.GaussianBlur(radius=1.4))
    low_quality.save(SAMPLE_DOCS / "receipt_low_quality.png", optimize=True)

    (SAMPLE_OUTPUTS / "ocr_result.txt").write_text(
        SAMPLE_OCR_TEXT,
        encoding="utf-8",
    )
    (SAMPLE_OUTPUTS / "ocr_result.json").write_text(
        json.dumps(
            {
                "source_mode": "synthetic_fixture",
                "provenance": {
                    "fixture_type": "synthetic_fixture",
                    "input_file": "receipt_sample.png",
                    "created_by": "course generator",
                    "disclaimer": "현재 실행에서 OCR을 호출한 결과가 아닙니다.",
                },
                "regions": SAMPLE_OCR_RESULT,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (SAMPLE_OUTPUTS / "paddleocr_vl_result.md").write_text(
        SAMPLE_VLM_MARKDOWN,
        encoding="utf-8",
    )
    (SAMPLE_OUTPUTS / "paddleocr_vl_result.json").write_text(
        json.dumps(SAMPLE_VLM_RESULT, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (SAMPLE_OUTPUTS / "extracted_result.json").write_text(
        json.dumps(SAMPLE_RECEIPT, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (SAMPLE_OUTPUTS / "receipt_result.xlsx").write_bytes(
        receipt_to_xlsx_bytes(SAMPLE_RECEIPT)
    )
    (SAMPLE_OUTPUTS / "golden_receipt_ocr.txt").write_text(
        GOLDEN_RECEIPT_OCR_TEXT,
        encoding="utf-8",
    )
    golden_receipt = {
        **GOLDEN_RECEIPT,
        "provenance": {
            "fixture_type": "human_verified_transcription_fixture",
            "input_file": "taebaek_restaurant_2025_redacted.png",
            "input_sha256": (
                "19227c7298a16ee69bef2d7bed65826b8a1cba5389375e4ae77d02005362641f"
            ),
            "engine": "not_executed",
            "engine_version": "not_applicable",
            "target_technology": "PaddleOCR Korean",
            "recorded_at": "2026-07-28",
            "reviewer": "course maintainer",
            "disclaimer": "현재 실행에서 모델을 호출한 결과가 아닙니다.",
        },
    }
    (SAMPLE_OUTPUTS / "golden_receipt.json").write_text(
        json.dumps(golden_receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (SAMPLE_OUTPUTS / "golden_receipt_result.xlsx").write_bytes(
        receipt_to_xlsx_bytes(
            golden_receipt,
            source_text=GOLDEN_RECEIPT_OCR_TEXT,
            review_record={
                "decision": "APPROVED",
                "reviewer": "course maintainer",
                "reviewed_at": "2026-07-28T00:00:00+09:00",
                "note": "공개 비식별 원본과 대조한 교육용 정답",
            },
        )
    )

    fields = {
        "net_amount": {
            "raw_value": "69,094",
            "normalized_value": 69094,
            "evidence": "원본의 공급가액 행",
        },
        "tax_amount": {
            "raw_value": "6,906",
            "normalized_value": 6906,
            "evidence": "원본의 부가세 행",
        },
        "total_amount": {
            "raw_value": "76,000",
            "normalized_value": 76000,
            "evidence": "원본의 합계 행",
        },
    }
    pipeline_trace = {
        "source_document": "taebaek_restaurant_2025_redacted.png",
        "source_mode": "human_verified_transcription_fixture",
        "provenance": golden_receipt["provenance"],
        "schema_fields": list(fields),
        "raw_text": "공급가액 69,094 / 부가세 6,906 / 합계 76,000",
        "fields": fields,
        "validation": {
            "amount_math_ok": 69094 + 6906 == 76000,
            "evidence_present": True,
        },
        "routing_decision": "REVIEW",
        "routing_reason": "정책상 원본을 보는 사람 확인 필수",
        "human_decision": "APPROVED_AFTER_SOURCE_CHECK",
        "next_step": "7교시에서 receipt_result.xlsx 생성",
    }
    (SAMPLE_OUTPUTS / "receipt_pipeline_trace.json").write_text(
        json.dumps(pipeline_trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("합성 영수증과 출처가 표시된 준비 결과를 생성했습니다.")


if __name__ == "__main__":
    main()
