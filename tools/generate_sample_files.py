"""합성 영수증 이미지와 수업용 mock 결과를 재생성한다."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export import receipt_to_csv_bytes
from src.sample_data import (
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
        json.dumps(SAMPLE_OCR_RESULT, ensure_ascii=False, indent=2) + "\n",
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
    (SAMPLE_OUTPUTS / "validated_result.csv").write_bytes(
        receipt_to_csv_bytes(SAMPLE_RECEIPT)
    )

    technology_comparison = {
        "input_document": "taebaek_restaurant_2025_redacted.png",
        "example_label": "교육용 예시 — 실제 모델 실행 결과가 아님",
        "comparisons": [
            {
                "technology": "OCR",
                "input": "영수증 이미지 픽셀",
                "process": ["텍스트 영역 탐지", "문자 인식"],
                "output": "교육용 예시: 텍스트·좌표·신뢰도",
                "cannot_guarantee": "업무 의미와 금액의 정확성",
            },
            {
                "technology": "VLM",
                "input": "영수증 이미지와 추출 지시",
                "process": ["시각·배치 확인", "언어 관계 해석"],
                "output": "교육용 예시: 표·Markdown·초안 JSON",
                "cannot_guarantee": "관계와 값의 사실성",
            },
            {
                "technology": "Document AI",
                "input": "영수증과 업무 규칙",
                "process": ["처리기 선택", "스키마", "검증", "사람 확인"],
                "output": "검토 가능한 업무 데이터",
                "cannot_guarantee": "검토 없는 완전 자동 정확성",
            },
        ],
        "document_ai_workflow": [
            "입력 품질",
            "OCR·VLM·혼합",
            "업무 스키마",
            "규칙 검증",
            "사람 확인",
            "저장",
        ],
    }
    (SAMPLE_OUTPUTS / "technology_comparison.json").write_text(
        json.dumps(technology_comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("합성 영수증과 mock 결과를 생성했습니다.")


if __name__ == "__main__":
    main()
