"""공개 한국 영수증 위에 문서 구조 학습용 영역을 표시한다."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "sample_docs"
    / "public_receipts"
    / "korea"
    / "taebaek_restaurant_2025_redacted.png"
)
OUTPUT = ROOT / "lessons" / "assets" / "03" / "03_receipt_regions.png"


def korean_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError("한글 글꼴을 찾을 수 없습니다.")


def main() -> None:
    source = Image.open(SOURCE).convert("RGBA")
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    label_font = korean_font(58)
    note_font = korean_font(38)

    regions = [
        ("헤더 · 상호명", (120, 240, 2450, 825), "#0F766E"),
        ("날짜 · 키-값", (120, 850, 2450, 1190), "#2563EB"),
        ("품목 · 반복 행", (120, 1200, 2450, 2110), "#D97706"),
        ("합계 · 중요 필드", (120, 2115, 2450, 2790), "#DC2626"),
    ]
    for label, box, color in regions:
        rgba = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
        draw.rounded_rectangle(
            box,
            radius=28,
            fill=(*rgba, 18),
            outline=(*rgba, 255),
            width=11,
        )
        text_box = draw.textbbox((0, 0), label, font=label_font)
        tag_width = text_box[2] - text_box[0] + 60
        tag_height = text_box[3] - text_box[1] + 44
        tag = (box[0] + 24, box[1] + 24, box[0] + 24 + tag_width, box[1] + 24 + tag_height)
        draw.rounded_rectangle(tag, radius=22, fill=(*rgba, 245))
        draw.text(
            (tag[0] + 30, tag[1] + 14),
            label,
            font=label_font,
            fill="white",
        )

    annotated = Image.alpha_composite(source, overlay).convert("RGB")
    target_width = 1100
    target_height = round(annotated.height * target_width / annotated.width)
    annotated = annotated.resize((target_width, target_height), Image.Resampling.LANCZOS)

    footer_height = 110
    canvas = Image.new("RGB", (target_width, target_height + footer_height), "#173B57")
    canvas.paste(annotated, (0, 0))
    footer = ImageDraw.Draw(canvas)
    footer.text(
        (target_width // 2, target_height + footer_height // 2),
        "교육용 문서 구조 주석 · 실제 OCR 검출 bbox가 아닙니다.",
        font=note_font,
        fill="white",
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)
    print(f"생성: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
