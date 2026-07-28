"""같은 한국 영수증을 1~7교시까지 이어 쓰는 Colab 노트북 생성기."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from textwrap import dedent

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"
GOLDEN_IMAGE = (
    ROOT
    / "sample_docs"
    / "public_receipts"
    / "korea"
    / "taebaek_restaurant_2025_redacted.png"
)
EXTENSION_IMAGES = {
    "quotation": ROOT / "sample_docs" / "extensions" / "quotation_photo.png",
    "application": ROOT
    / "sample_docs"
    / "extensions"
    / "application_form_photo.png",
    "transaction_statement": ROOT
    / "sample_docs"
    / "extensions"
    / "transaction_statement_photo.png",
}

GOLDEN_OCR_TEXT = """이태리집
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
    "adjustments": {"discount": 0, "tax": 0, "service": 0, "rounding": 0},
    "tax_breakdown": {
        "mode": "included_in_item_prices",
        "supply_amount": 69094,
        "vat": 6906,
        "payable_total": 76000,
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
    "evidence": {
        "store_name": {"raw_value": "이태리집", "line": 1},
        "date": {"raw_value": "거래일시 2025-10-04 12:33:37", "line": 2},
        "total_amount": {"raw_value": "합계 금액 76,000", "line": 8},
    },
    "source_mode": "prepared_fixture_rule_extraction",
}

EXTENSION_EXAMPLES = {
    "quotation": {
        "name": "견적서",
        "fields": ["문서번호", "공급자", "수신", "견적일", "품목", "총액"],
        "rules": ["수량×단가=품목금액", "공급가액+부가세=총액"],
        "risk": "총액 오류는 구매 의사결정에 직접 영향",
    },
    "application": {
        "name": "신청서",
        "fields": ["신청번호", "신청자", "소속", "신청 과정", "승인"],
        "rules": ["필수 동의", "관리자 승인 상태"],
        "risk": "개인정보와 승인 누락을 사람이 확인",
    },
    "transaction_statement": {
        "name": "거래명세서",
        "fields": ["문서번호", "공급자", "거래일", "품목", "세액", "총액"],
        "rules": ["품목 합계=공급가액", "공급가액+세액=총액"],
        "risk": "표 행·열 대응이 어긋나면 정산 오류",
    },
}

NOTEBOOK_SLUGS = {
    1: "document_ai_overview",
    2: "ocr_basic",
    3: "document_structure",
    4: "genai_extraction",
    5: "streamlit_basic",
    6: "ocr_ai_integration",
    7: "validation_export",
    8: "business_application",
}


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": "pending",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "pending",
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def notebook(name: str, cells: list[dict]) -> dict:
    prefix = name[:2]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"{prefix}-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def intro(lesson: int, title: str, artifact: str, goal: str) -> dict:
    slug = NOTEBOOK_SLUGS[lesson]
    return markdown(
        f"""
        # {lesson}교시. {title}

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/{lesson:02d}_{slug}.ipynb)

        **이번 교시 행동:** {goal}

        **통과 증거:** `course_outputs/{artifact}`

        > Google Colab도 외부 클라우드입니다. 조직 승인 없는 개인·회사 문서는
        > 업로드하지 않습니다. 필수 실습은 저장소의 비식별 공개·합성 샘플만
        > 사용합니다.

        화면의 실행 모드를 먼저 확인합니다.

        - `LIVE`: 현재 파일에 실제 모델을 실행한 결과
        - `PREPARED_FALLBACK`: 공개 샘플을 사람이 검수해 둔 복구 결과
        - 3분 이상 멈추면 실행을 중지하고 복구 결과로 계속합니다.
        - 각 교시 끝에서 `CHECKPOINT PASS`와 산출물 파일을 확인합니다.
        """
    )


def runtime_cell() -> dict:
    return code(
        """
        import json
        import os
        import platform
        import sys
        from pathlib import Path

        OUTPUT_DIR = Path("course_outputs")
        OUTPUT_DIR.mkdir(exist_ok=True)
        VALIDATION_MODE = os.getenv("COURSE_VALIDATE_PREPARED") == "1"

        def upload_previous_artifact(filename):
            target = OUTPUT_DIR / filename
            if target.exists() or VALIDATION_MODE:
                return target if target.exists() else None
            try:
                from google.colab import files
            except ImportError:
                return None
            print(f"이전 교시에서 내려받은 {filename}을 선택하세요.")
            uploaded = files.upload()
            if filename not in uploaded:
                raise FileNotFoundError(
                    f"{filename}이 선택되지 않았습니다. 준비 입력을 쓰려면 "
                    "USE_PREPARED_INPUT=True로 바꾸세요."
                )
            target.write_bytes(uploaded[filename])
            return target


        def download_artifact(path):
            if VALIDATION_MODE:
                return
            try:
                from google.colab import files
            except ImportError:
                return
            files.download(str(path))

        print("Python:", sys.version.split()[0])
        print("Platform:", platform.platform())
        print("공통 작업 폴더:", OUTPUT_DIR.resolve())
        """
    )


def image_base64(path: Path, max_size: tuple[int, int] = (900, 1100)) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(max_size)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def golden_constants() -> str:
    return (
        f"GOLDEN_OCR_TEXT = {GOLDEN_OCR_TEXT!r}\n"
        f"GOLDEN_RECEIPT = {GOLDEN_RECEIPT!r}\n"
    )


def parser_source() -> str:
    return r'''
import re

def to_int(value):
    return int(value.replace(",", ""))


def extract_receipt_from_text(text, source_mode):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    date_match = re.search(r"\b(\d{4})[-./](\d{1,2})[-./](\d{1,2})\b", text)
    total_line = next(
        (
            line
            for line in lines
            if re.search(r"(?:합\s*계|결제\s*금액|총\s*액)", line)
        ),
        None,
    )
    total_candidates = (
        re.findall(r"(?<![\d,])\d[\d,]*(?![\d,])", total_line)
        if total_line
        else []
    )
    total_raw = total_candidates[-1] if total_candidates else None
    supply_match = re.search(
        r"(?:부가세\s*)?과세물품가액\s*[:：]?\s*([\d,]+)",
        text,
    )
    vat_match = re.search(
        r"^부가세(?!\s*과세물품가액)\s*[:：]?\s*([\d,]+)",
        text,
        re.MULTILINE,
    )
    item_pattern = re.compile(
        r"^(?P<name>.+?)\s+(?P<unit>[\d,]+)\s+"
        r"(?P<quantity>\d+)\s+(?P<line>[\d,]+)$"
    )
    items = []
    item_evidence = []
    for line_number, line in enumerate(lines, start=1):
        match = item_pattern.search(line)
        if match:
            item = {
                "name": match.group("name"),
                "quantity": int(match.group("quantity")),
                "unit_price": to_int(match.group("unit")),
                "line_total": to_int(match.group("line")),
            }
            items.append(item)
            item_evidence.append({"line": line_number, "raw_value": line})

    date_value = (
        f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-"
        f"{int(date_match.group(3)):02d}"
        if date_match else None
    )
    total_value = to_int(total_raw) if total_raw else None
    supply_value = to_int(supply_match.group(1)) if supply_match else None
    vat_value = to_int(vat_match.group(1)) if vat_match else None
    return {
        "document_type": "receipt",
        "store_name": lines[0] if lines else None,
        "date": date_value,
        "total_amount": total_value,
        "items": items,
        "adjustments": {"discount": 0, "tax": 0, "service": 0, "rounding": 0},
        "tax_breakdown": {
            "mode": "included_in_item_prices",
            "supply_amount": supply_value,
            "vat": vat_value,
            "payable_total": total_value,
        } if supply_value is not None and vat_value is not None else None,
        "raw_values": {
            "store_name": lines[0] if lines else None,
            "date": date_match.group(0) if date_match else None,
            "total_amount": total_raw,
        },
        "cleaned_values": {
            "store_name": lines[0] if lines else None,
            "date": date_value,
            "total_amount": total_value,
        },
        "evidence": {
            "store_name": {"line": 1, "raw_value": lines[0] if lines else None},
            "date": {"raw_value": date_match.group(0) if date_match else None},
            "total_amount": {"raw_value": total_line},
            "items": item_evidence,
        },
        "source_mode": source_mode,
    }
'''


def notebook_01() -> dict:
    encoded = image_base64(GOLDEN_IMAGE)
    cells = [
        intro(
            1,
            "한국 영수증으로 구분하는 OCR·VLM·Document AI",
            "receipt_pipeline_trace.json",
            "실제 영수증과 최종 Excel 사이의 역할 네 가지를 직접 연결합니다.",
        ),
        runtime_cell(),
        code(
            f"""
            import base64
            import io
            from PIL import Image

            GOLDEN_IMAGE_BASE64 = {encoded!r}
            receipt_image = Image.open(
                io.BytesIO(base64.b64decode(GOLDEN_IMAGE_BASE64))
            ).convert("RGB")
            receipt_image.thumbnail((720, 900))
            receipt_image
            """
        ),
        markdown(
            """
            ## 기억할 네 문장

            - **OCR**은 문서 이미지에서 글자와 위치를 읽습니다.
            - **VLM**은 이미지와 언어를 함께 보고 문서의 의미·구조 초안을 만듭니다.
            - **Document AI**는 분류·읽기·구조화·검증을 문서 처리 능력으로 묶습니다.
            - **IDP**는 사람 승인, 예외 처리, 업무 시스템 연결, 운영 개선까지 포함합니다.

            이 네 용어는 경쟁 제품 이름이 아니라 **포함 범위가 넓어지는 관계**입니다.
            `OCR → VLM`은 반드시 거치는 고정 순서가 아닙니다.
            """
        ),
        code(
            f"""
            pipeline_trace = {{
                "source_document": "taebaek_restaurant_2025_redacted.png",
                "input_policy": "approved_redacted_public_sample",
                "roles": [
                    {{"role": "OCR", "action": "글자·위치 판독", "artifact": "ocr_result.json"}},
                    {{"role": "VLM", "action": "문서 의미·구조 초안", "artifact": "receipt_draft.json"}},
                    {{"role": "Document AI", "action": "스키마·근거·규칙 검증", "artifact": "validated_receipt.json"}},
                    {{"role": "IDP", "action": "사람 승인 후 Excel 연결", "artifact": "receipt_result.xlsx"}},
                ],
                "evidence_example": {{
                    "field": "total_amount",
                    "value": 76000,
                    "source_text": "합계 금액 76,000",
                    "decision": "REVIEW_BEFORE_EXPORT",
                }},
                "scope_note": "0~12 전체 지도는 참고용이며 오늘은 한 장 경로를 구현합니다.",
            }}
            output_path = OUTPUT_DIR / "receipt_pipeline_trace.json"
            output_path.write_text(
                json.dumps(pipeline_trace, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            assert len(pipeline_trace["roles"]) == 4
            print("CHECKPOINT 1/1 PASS:", output_path)
            """
        ),
    ]
    return notebook("01_document_ai_overview.ipynb", cells)


def notebook_02() -> dict:
    encoded = image_base64(GOLDEN_IMAGE)
    prepared = [
        {
            "box": [[48, 105], [360, 105], [360, 155], [48, 155]],
            "text": "이태리집",
            "confidence": None,
        },
        {
            "box": [[48, 345], [650, 345], [650, 395], [48, 395]],
            "text": "거래일시 2025-10-04 12:33:37",
            "confidence": None,
        },
        {
            "box": [[48, 475], [820, 475], [820, 515], [48, 515]],
            "text": "페퍼로니 앤 치즈 29,000 1 29,000",
            "confidence": None,
        },
        {
            "box": [[48, 515], [820, 515], [820, 555], [48, 555]],
            "text": "토마토 파스타 14,000 1 14,000",
            "confidence": None,
        },
        {
            "box": [[48, 555], [820, 555], [820, 595], [48, 595]],
            "text": "수제 돈가스 13,000 1 13,000",
            "confidence": None,
        },
        {
            "box": [[48, 595], [820, 595], [820, 635], [48, 635]],
            "text": "새우 칠리치 필라 14,000 1 14,000",
            "confidence": None,
        },
        {
            "box": [[48, 635], [820, 635], [820, 675], [48, 675]],
            "text": "콜라 2,000 3 6,000",
            "confidence": None,
        },
        {
            "box": [[48, 790], [820, 790], [820, 840], [48, 840]],
            "text": "합계 금액 76,000",
            "confidence": None,
        },
        {
            "box": [[48, 850], [820, 850], [820, 890], [48, 890]],
            "text": "부가세 과세물품가액 69,094",
            "confidence": None,
        },
        {
            "box": [[48, 890], [820, 890], [820, 930], [48, 930]],
            "text": "부가세 6,906",
            "confidence": None,
        },
    ]
    cells = [
        intro(
            2,
            "OCR 기반 텍스트 추출 실습",
            "ocr_result.json",
            "공개 한국 영수증에 실제 OCR을 실행하고 원본 위 좌표·신뢰도를 확인합니다.",
        ),
        runtime_cell(),
        code(
            f"""
            import base64
            import io
            from PIL import Image, ImageDraw

            GOLDEN_IMAGE_BASE64 = {encoded!r}
            receipt_image = Image.open(
                io.BytesIO(base64.b64decode(GOLDEN_IMAGE_BASE64))
            ).convert("RGB")
            PREPARED_OCR_RESULT = {prepared!r}
            for item in PREPARED_OCR_RESULT:
                item["confidence_source"] = "not_available_prepared_fixture"
            RUN_LIVE_OCR = not VALIDATION_MODE
            print("요청 모드:", "LIVE" if RUN_LIVE_OCR else "PREPARED_FALLBACK")
            """
        ),
        markdown(
            """
            ## 실행

            기본값은 `LIVE`입니다. Colab에서 PaddleOCR 3.7과
            `PP-OCRv5 Korean`을 사용합니다. PP-OCRv6가 최신 기본 계열이어도
            한국어 전용 인식 모델은 PP-OCRv5 Korean을 사용합니다.

            설치·모델 다운로드가 3분을 넘기면 중지합니다. 오류 메시지를 보존한
            채 `PREPARED_FALLBACK`으로 전환하며, 전환 사실을 결과 JSON에 남깁니다.
            """
        ),
        code(
            """
            OCR_RESULT = PREPARED_OCR_RESULT
            SOURCE_MODE = "PREPARED_FALLBACK"
            FALLBACK_REASON = "offline validator"

            if RUN_LIVE_OCR:
                import subprocess
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-q",
                         "paddlepaddle==3.2.1", "paddleocr==3.7.0"]
                    )
                    from paddleocr import PaddleOCR

                    image_path = OUTPUT_DIR / "golden_receipt.jpg"
                    receipt_image.save(image_path)
                    engine = PaddleOCR(
                        lang="korean",
                        ocr_version="PP-OCRv5",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        device="cpu",
                    )
                    page = list(engine.predict(str(image_path)))[0]
                    payload = page.json() if callable(page.json) else page.json
                    result = payload.get("res", payload)
                    OCR_RESULT = [
                        {
                            "box": box,
                            "text": text,
                            "confidence": float(score),
                        }
                        for box, text, score in zip(
                            result.get("rec_polys", []),
                            result.get("rec_texts", []),
                            result.get("rec_scores", []),
                        )
                    ]
                    SOURCE_MODE = "LIVE"
                    FALLBACK_REASON = ""
                except Exception as exc:
                    SOURCE_MODE = "PREPARED_FALLBACK"
                    FALLBACK_REASON = f"{type(exc).__name__}: {exc}"

            print("실행 모드:", SOURCE_MODE)
            if FALLBACK_REASON:
                print("복구 사유:", FALLBACK_REASON)
            print("판독 영역:", len(OCR_RESULT))
            """
        ),
        code(
            """
            annotated = receipt_image.copy()
            draw = ImageDraw.Draw(annotated)
            scale_x = annotated.width / 900
            scale_y = annotated.height / 1100
            for item in OCR_RESULT:
                points = item["box"]
                xs = [point[0] * scale_x for point in points]
                ys = [point[1] * scale_y for point in points]
                draw.rectangle(
                    (min(xs), min(ys), max(xs), max(ys)),
                    outline="#0F766E",
                    width=4,
                )
            annotated_path = OUTPUT_DIR / "ocr_boxes.png"
            annotated.save(annotated_path)

            output = {
                "source_mode": SOURCE_MODE,
                "fallback_reason": FALLBACK_REASON,
                "input_file": "taebaek_restaurant_2025_redacted.png",
                "items": [
                    {**item, "matches_source": None, "review_note": ""}
                    for item in OCR_RESULT
                ],
            }
            output_path = OUTPUT_DIR / "ocr_result.json"
            output_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            print("CHECKPOINT 1/1 PASS:", SOURCE_MODE, output_path, annotated_path)
            download_artifact(output_path)
            download_artifact(annotated_path)
            """
        ),
    ]
    return notebook("02_ocr_basic.ipynb", cells)


def notebook_03() -> dict:
    cells = [
        intro(
            3,
            "문서 구조 이해 및 추출 결과 정제",
            "clean_receipt.json",
            "2교시 결과를 불러와 원문은 보존하고, 공백·날짜·표 영역만 정리합니다.",
        ),
        runtime_cell(),
        code(golden_constants()),
        code(
            """
            import re

            def reconstruct_spatial_lines(items):
                tokens = []
                unpositioned = []
                for order, item in enumerate(items):
                    text = re.sub(r"\\s+", " ", item.get("text", "").strip())
                    if not text:
                        continue
                    box = item.get("box") or []
                    points = [
                        point
                        for point in box
                        if isinstance(point, (list, tuple)) and len(point) >= 2
                    ]
                    if not points:
                        unpositioned.append(text)
                        continue
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                    tokens.append({
                        "text": text,
                        "x": min(xs),
                        "y": sum(ys) / len(ys),
                        "height": max(ys) - min(ys),
                        "order": order,
                    })

                rows = []
                for token in sorted(tokens, key=lambda value: (value["y"], value["x"])):
                    if rows:
                        row = rows[-1]
                        tolerance = max(
                            12.0,
                            min(24.0, max(row["height"], token["height"]) * 0.45),
                        )
                    else:
                        row = None
                        tolerance = 12.0
                    if row and abs(token["y"] - row["y"]) <= tolerance:
                        row["tokens"].append(token)
                        count = len(row["tokens"])
                        row["y"] = (
                            row["y"] * (count - 1) + token["y"]
                        ) / count
                        row["height"] = max(row["height"], token["height"])
                    else:
                        rows.append({
                            "tokens": [token],
                            "y": token["y"],
                            "height": token["height"],
                        })

                spatial_lines = [
                    " ".join(
                        token["text"]
                        for token in sorted(row["tokens"], key=lambda value: value["x"])
                    )
                    for row in rows
                ]
                return spatial_lines + unpositioned


            previous_path = OUTPUT_DIR / "ocr_result.json"
            USE_PREPARED_INPUT = VALIDATION_MODE
            if not previous_path.exists() and not USE_PREPARED_INPUT:
                upload_previous_artifact("ocr_result.json")
            if previous_path.exists():
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                raw_text = "\\n".join(item["text"] for item in previous["items"])
                layout_lines = reconstruct_spatial_lines(previous["items"])
                INPUT_MODE = "PREVIOUS_LESSON"
            else:
                raw_text = GOLDEN_OCR_TEXT
                layout_lines = raw_text.splitlines()
                INPUT_MODE = "PREPARED_FALLBACK"
            print("입력 모드:", INPUT_MODE)


            def clean_lines(text, source_lines):
                cleaned = []
                changes = []
                for raw in source_lines:
                    normalized = re.sub(r"\\s+", " ", raw.strip())
                    if normalized:
                        cleaned.append(normalized)
                    if raw != normalized:
                        changes.append({"before": raw, "after": normalized})
                groups = {"header": [], "date": [], "items": [], "total": [], "other": []}
                for line in cleaned:
                    if re.search(r"\\d{4}[-./]\\d{1,2}[-./]\\d{1,2}", line):
                        groups["date"].append(line)
                    elif "합계" in line:
                        groups["total"].append(line)
                    elif re.search(r"[\\d,]+\\s+\\d+\\s+[\\d,]+$", line):
                        groups["items"].append(line)
                    elif not groups["header"]:
                        groups["header"].append(line)
                    else:
                        groups["other"].append(line)
                return {
                    "input_mode": INPUT_MODE,
                    "raw_text": text,
                    "layout_lines": source_lines,
                    "cleaned_lines": cleaned,
                    "groups": groups,
                    "change_log": changes,
                    "rule": "원문에 없는 값은 추가하지 않음",
                }


            clean_result = clean_lines(raw_text, layout_lines)
            output_path = OUTPUT_DIR / "clean_receipt.json"
            output_path.write_text(
                json.dumps(clean_result, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            assert clean_result["raw_text"] == raw_text
            print("원문 줄:", len(raw_text.splitlines()))
            print("품목 후보 줄:", len(clean_result["groups"]["items"]))
            print("CHECKPOINT 1/1 PASS:", output_path)
            download_artifact(output_path)
            """
        ),
    ]
    return notebook("03_document_structure.ipynb", cells)


def notebook_04() -> dict:
    cells = [
        intro(
            4,
            "멀티모달·생성형 AI 기반 핵심 정보 추출",
            "receipt.json",
            "같은 영수증을 업무 JSON 초안으로 만들고 모든 핵심값에 원본 근거를 붙입니다.",
        ),
        runtime_cell(),
        code(golden_constants()),
        code(parser_source()),
        markdown(
            """
            ## VLM 결과를 정답으로 보지 않는 세 가지 확인

            1. **스키마**: 필요한 필드와 자료형이 맞는가?
            2. **근거**: 값이 원본 어느 줄에서 왔는가?
            3. **불확실성**: 근거가 없으면 추측하지 않고 `null`인가?

            이번 필수 경로는 비용과 네트워크 변수를 없애기 위해 사람이 검수한
            준비 텍스트를 사용합니다. 실제 VLM 호출 결과로 오해하지 않도록
            provenance를 함께 저장합니다.
            """
        ),
        code(
            """
            previous_path = OUTPUT_DIR / "clean_receipt.json"
            USE_PREPARED_INPUT = VALIDATION_MODE
            if not previous_path.exists() and not USE_PREPARED_INPUT:
                upload_previous_artifact("clean_receipt.json")
            if previous_path.exists():
                clean_result = json.loads(previous_path.read_text(encoding="utf-8"))
                source_text = "\\n".join(clean_result["cleaned_lines"])
                INPUT_MODE = "PREVIOUS_LESSON"
            else:
                source_text = GOLDEN_OCR_TEXT
                INPUT_MODE = "PREPARED_FALLBACK"

            receipt = extract_receipt_from_text(
                source_text,
                "prepared_vlm_draft_with_rule_normalization",
            )
            receipt["provenance"] = {
                "fixture_type": "human_verified_transcription_fixture",
                "input_file": "taebaek_restaurant_2025_redacted.png",
                "input_sha256": "19227c7298a16ee69bef2d7bed65826b8a1cba5389375e4ae77d02005362641f",
                "engine": "not_executed",
                "engine_version": "not_applicable",
                "target_technology": "PaddleOCR-VL 1.6 output structure",
                "recorded_at": "2026-07-28",
                "reviewer": "course maintainer",
                "disclaimer": "현재 실행에서 VLM을 호출한 결과가 아닙니다.",
            }
            receipt["input_mode"] = INPUT_MODE
            assert receipt["total_amount"] == 76000
            assert receipt["evidence"]["total_amount"]["raw_value"]
            output_path = OUTPUT_DIR / "receipt.json"
            output_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "total_amount": receipt["total_amount"],
                "evidence": receipt["evidence"]["total_amount"],
                "source_mode": receipt["source_mode"],
            }, ensure_ascii=False, indent=2))
            print("CHECKPOINT 1/1 PASS:", output_path)
            download_artifact(output_path)
            """
        ),
    ]
    return notebook("04_genai_extraction.ipynb", cells)


STREAMLIT_SETUP = """
import importlib.metadata
import subprocess

required_streamlit = "1.60.0"
try:
    installed_streamlit = importlib.metadata.version("streamlit")
except importlib.metadata.PackageNotFoundError:
    installed_streamlit = None
if installed_streamlit != required_streamlit:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", f"streamlit=={required_streamlit}"]
    )
"""


def notebook_05() -> dict:
    app_source = dedent(
        f'''
        import streamlit as st

        GOLDEN_RECEIPT = {GOLDEN_RECEIPT!r}
        GOLDEN_OCR_TEXT = {GOLDEN_OCR_TEXT!r}

        st.set_page_config(page_title="영수증 Document AI", layout="wide")
        st.title("영수증 Document AI 미니 앱")
        uploaded = st.file_uploader(
            "승인된 비식별 이미지 또는 PDF 한 장 · 최대 5MB",
            type=["png", "jpg", "jpeg", "pdf"],
            max_upload_size=5,
            help="PNG, JPG, JPEG, PDF만 허용합니다. 수업에서는 한 번에 5MB 이하 한 장만 처리합니다.",
        )
        if uploaded is not None:
            st.success(f"업로드 연결 확인: {{uploaded.name}} · {{len(uploaded.getvalue()):,}} bytes")
            st.caption("이 파일은 6교시에서 실제 처리 함수와 연결합니다.")

        if st.button("공개 샘플 준비 결과 보기"):
            st.info("실행 모드: PREPARED_FALLBACK")
            st.text_area("판독 원문", GOLDEN_OCR_TEXT, height=220)
            st.json(GOLDEN_RECEIPT)
        '''
    ).lstrip()
    cells = [
        intro(
            5,
            "문서 자동화 웹 애플리케이션 기본 구현",
            "app_05.py",
            "업로드·실행·원문·JSON 영역을 만들고 업로드한 파일명이 화면에 반영되는지 확인합니다.",
        ),
        runtime_cell(),
        code(STREAMLIT_SETUP),
        code(
            f"""
            from textwrap import dedent

            app_code = {app_source!r}
            output_path = OUTPUT_DIR / "app_05.py"
            output_path.write_text(app_code, encoding="utf-8")
            print("저장:", output_path)
            """
        ),
        code(
            """
            from streamlit.testing.v1 import AppTest

            app_test = AppTest.from_file(str(output_path)).run(timeout=20)
            assert not app_test.exception
            assert app_test.title[0].value == "영수증 Document AI 미니 앱"
            assert len(app_test.file_uploader) == 1
            assert len(app_test.button) == 1
            app_test.button[0].click().run(timeout=20)
            assert any("PREPARED_FALLBACK" in item.value for item in app_test.info)
            print("CHECKPOINT 1/1 PASS: 업로드·버튼·결과 화면")
            """
        ),
    ]
    return notebook("05_streamlit_basic.ipynb", cells)


def notebook_06() -> dict:
    app_source = (
        dedent(
            f'''
            import tempfile
            from pathlib import Path
            import streamlit as st

            GOLDEN_OCR_TEXT = {GOLDEN_OCR_TEXT!r}
            GOLDEN_RECEIPT = {GOLDEN_RECEIPT!r}
            '''
        ).lstrip()
        + "\n"
        + parser_source().strip()
        + "\n\n"
        + dedent(
        f'''
        def run_live_ocr(uploaded):
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(uploaded.getvalue())
                path = temp.name
            try:
                from paddleocr import PaddleOCR
                engine = PaddleOCR(
                    lang="korean",
                    ocr_version="PP-OCRv5",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                )
                page = list(engine.predict(path))[0]
                payload = page.json() if callable(page.json) else page.json
                result = payload.get("res", payload)
                return "\\n".join(result.get("rec_texts", []))
            finally:
                Path(path).unlink(missing_ok=True)


        def process_document(uploaded=None, *, use_prepared=False):
            if use_prepared:
                text = GOLDEN_OCR_TEXT
                mode = "PREPARED_FALLBACK"
            elif uploaded is None:
                return {{"ok": False, "mode": "INPUT_ERROR", "error": "파일을 선택하세요."}}
            else:
                try:
                    text = run_live_ocr(uploaded)
                    mode = "LIVE"
                except Exception as exc:
                    return {{
                        "ok": False,
                        "mode": "LIVE_ERROR",
                        "error": f"{{type(exc).__name__}}: {{exc}}",
                        "recovery": "공개 샘플 준비 결과 버튼을 선택하세요.",
                    }}
            data = extract_receipt_from_text(
                text,
                "live_ocr_rule_extraction" if mode == "LIVE"
                else "prepared_fixture_rule_extraction",
            )
            return {{"ok": True, "mode": mode, "ocr_text": text, "data": data}}


        st.title("영수증 Document AI 연결 앱")
        uploaded = st.file_uploader(
            "승인된 비식별 이미지 또는 PDF 한 장 · 최대 5MB",
            type=["png", "jpg", "jpeg", "pdf"],
            max_upload_size=5,
            help="PNG, JPG, JPEG, PDF만 허용합니다. 수업에서는 한 번에 5MB 이하 한 장만 처리합니다.",
        )
        left, right = st.columns(2)
        run_live = left.button("업로드 파일 LIVE 처리")
        run_prepared = right.button("공개 샘플 준비 결과")
        if run_live or run_prepared:
            result = process_document(uploaded, use_prepared=run_prepared)
            if result["ok"]:
                st.success(f"실행 모드: {{result['mode']}}")
                st.text_area("OCR 원문", result["ocr_text"], height=220)
                st.json(result["data"])
            else:
                st.error(f"{{result['mode']}} · {{result['error']}}")
                if result.get("recovery"):
                    st.info(result["recovery"])
        '''
        ).lstrip()
    )
    cells = [
        intro(
            6,
            "OCR 및 정보 추출 기능 연동",
            "app_06.py",
            "업로드한 파일을 실제 OCR 함수에 연결하고 LIVE·오류·복구 모드를 화면에서 구분합니다.",
        ),
        runtime_cell(),
        code(STREAMLIT_SETUP),
        code(
            f"""
            from textwrap import dedent

            app_code = {app_source!r}
            output_path = OUTPUT_DIR / "app_06.py"
            output_path.write_text(app_code, encoding="utf-8")
            print("저장:", output_path)
            """
        ),
        code(
            """
            from streamlit.testing.v1 import AppTest

            app_test = AppTest.from_file(str(output_path)).run(timeout=20)
            assert not app_test.exception
            assert app_test.title[0].value == "영수증 Document AI 연결 앱"
            assert len(app_test.button) == 2
            app_test.button[1].click().run(timeout=20)
            assert any("PREPARED_FALLBACK" in item.value for item in app_test.success)
            assert app_test.json
            print("CHECKPOINT 1/1 PASS: 앱 연결·모드 표시·JSON 출력")
            """
        ),
    ]
    return notebook("06_ocr_ai_integration.ipynb", cells)


def notebook_07() -> dict:
    cells = [
        intro(
            7,
            "추출 결과 검증 및 데이터 저장",
            "receipt_result.xlsx",
            "오류·경고·사람 검토를 분리하고, 공개된 승인 정답 경로에서만 Excel을 만듭니다.",
        ),
        runtime_cell(),
        code(
            """
            import importlib.util
            import subprocess
            if importlib.util.find_spec("openpyxl") is None:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "openpyxl==3.1.5"]
                )
            from copy import deepcopy
            from datetime import date
            from openpyxl import Workbook, load_workbook
            """
        ),
        code(golden_constants()),
        code(
            """
            input_path = OUTPUT_DIR / "receipt.json"
            USE_PREPARED_INPUT = VALIDATION_MODE
            if not input_path.exists() and not USE_PREPARED_INPUT:
                upload_previous_artifact("receipt.json")
            if input_path.exists():
                receipt = json.loads(input_path.read_text(encoding="utf-8"))
                INPUT_MODE = "PREVIOUS_LESSON"
            else:
                receipt = deepcopy(GOLDEN_RECEIPT)
                INPUT_MODE = "PREPARED_FALLBACK"


            def validate_receipt(data):
                warnings, errors = [], []
                for field in ("store_name", "date", "total_amount", "items"):
                    if data.get(field) in (None, "", []):
                        errors.append(f"필수값 누락: {field}")
                try:
                    parsed_date = date.fromisoformat(data.get("date", ""))
                    if parsed_date > date.today():
                        warnings.append("미래 날짜입니다. 원본을 확인하세요.")
                except ValueError:
                    errors.append("date는 YYYY-MM-DD 형식이어야 합니다.")

                total = data.get("total_amount")
                if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                    errors.append("total_amount는 0 이상의 정수여야 합니다.")
                item_sum = 0
                for index, item in enumerate(data.get("items") or [], start=1):
                    values = [item.get(key) for key in ("quantity", "unit_price", "line_total")]
                    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                        errors.append(f"{index}번째 품목 금액 형식 오류")
                        continue
                    if values[0] * values[1] != values[2]:
                        errors.append(f"{index}번째 품목 수량×단가 오류")
                    item_sum += values[2]
                adjustments = data.get("adjustments") or {}
                expected = (
                    item_sum
                    - adjustments.get("discount", 0)
                    + adjustments.get("tax", 0)
                    + adjustments.get("service", 0)
                    + adjustments.get("rounding", 0)
                )
                if isinstance(total, int) and not isinstance(total, bool) and expected != total:
                    errors.append(f"품목·조정 후 합계 {expected:,}원과 총액 {total:,}원이 다릅니다.")
                tax_breakdown = data.get("tax_breakdown")
                if tax_breakdown and tax_breakdown.get("mode") == "included_in_item_prices":
                    supply = tax_breakdown.get("supply_amount")
                    vat = tax_breakdown.get("vat")
                    payable = tax_breakdown.get("payable_total")
                    if not all(isinstance(value, int) and not isinstance(value, bool)
                               for value in (supply, vat, payable)):
                        errors.append("포함세액 내역은 정수 금액이어야 합니다.")
                    elif supply + vat != payable or payable != total:
                        errors.append("공급가액·포함 부가세·총액 관계가 맞지 않습니다.")
                    if adjustments.get("tax", 0) != 0:
                        errors.append("포함 부가세를 adjustments.tax에 다시 더하면 이중 계산됩니다.")
                for field in ("store_name", "date", "total_amount"):
                    if not (data.get("evidence") or {}).get(field):
                        warnings.append(f"{field}의 원본 근거가 없습니다.")
                return {"valid": not errors, "warnings": warnings, "errors": errors}


            validation = validate_receipt(receipt)
            assert validation["valid"], validation
            print("입력 모드:", INPUT_MODE)
            print("검증:", validation)
            """
        ),
        code(
            """
            def safe_text(value):
                if isinstance(value, str) and value.lstrip(" \\t\\r\\n").startswith(
                    ("=", "+", "-", "@")
                ):
                    return "'" + value
                return value


            def save_reviewed_excel(data, validation, review_record, output_path, source_text):
                if not validation["valid"]:
                    return False
                if review_record.get("decision") not in {"APPROVED", "CHANGED"}:
                    return False

                workbook = Workbook()
                summary = workbook.active
                summary.title = "검토_요약"
                summary.append([
                    "field", "raw_value", "cleaned_value", "final_value",
                    "decision", "reviewer", "reviewed_at", "change_reason",
                ])
                raw = data.get("raw_values") or {}
                cleaned = data.get("cleaned_values") or {}
                for field in ("store_name", "date", "total_amount"):
                    summary.append([
                        field,
                        safe_text(raw.get(field)),
                        safe_text(cleaned.get(field)),
                        safe_text(data.get(field)),
                        review_record["decision"],
                        safe_text(review_record["reviewer"]),
                        review_record["reviewed_at"],
                        safe_text(review_record["note"]),
                    ])

                items = workbook.create_sheet("품목")
                items.append(["품목", "수량", "단가", "금액"])
                for item in data["items"]:
                    items.append([
                        safe_text(item["name"]),
                        item["quantity"],
                        item["unit_price"],
                        item["line_total"],
                    ])

                evidence = workbook.create_sheet("원문_근거")
                evidence.append(["source_mode", data.get("source_mode")])
                evidence.append(["ocr_text", safe_text(source_text)])
                evidence.append(["evidence", safe_text(json.dumps(
                    data.get("evidence") or {}, ensure_ascii=False
                ))])
                workbook.save(output_path)
                return True
            """
        ),
        markdown(
            """
            ## 시나리오 A. 기본값은 차단

            사람이 원본을 보기 전에는 결과가 유효해도 다운로드를 열지 않습니다.
            """
        ),
        code(
            """
            blocked_path = OUTPUT_DIR / "pending_review.xlsx"
            PENDING_REVIEW = {
                "decision": "PENDING",
                "reviewer": "",
                "reviewed_at": "",
                "note": "",
            }
            assert not save_reviewed_excel(
                receipt, validation, PENDING_REVIEW, blocked_path, GOLDEN_OCR_TEXT
            )
            assert not blocked_path.exists()
            print("DEFAULT_BLOCKED PASS: 미승인 Excel 없음")
            """
        ),
        markdown(
            """
            ## 시나리오 B. 전체 정답 공개 — 원본 확인 후 실행

            아래 셀은 승인 기록의 **완성 정답**입니다. 원본의 상호명·날짜·품목·총액을
            직접 대조한 뒤 실행합니다. 결정·검토자·시각·메모가 Excel에 남습니다.
            """
        ),
        code(
            """
            REVIEW_RECORD = {
                "decision": "APPROVED",
                "reviewer": "learner",
                "reviewed_at": "2026-07-28T15:30:00+09:00",
                "note": "공개 비식별 원본과 상호명·날짜·품목·총액 대조 완료",
            }
            output_path = OUTPUT_DIR / "receipt_result.xlsx"
            assert save_reviewed_excel(
                receipt, validation, REVIEW_RECORD, output_path, GOLDEN_OCR_TEXT
            )
            saved = load_workbook(output_path)
            assert saved.sheetnames == ["검토_요약", "품목", "원문_근거"]
            assert saved["검토_요약"]["E2"].value == "APPROVED"
            print("REVIEWED_APPROVED PASS:", output_path, saved.sheetnames)
            print("CHECKPOINT 1/1 PASS: 미승인 차단 + 승인 후 Excel")
            download_artifact(output_path)
            """
        ),
    ]
    return notebook("07_validation_export.ipynb", cells)


def notebook_08() -> dict:
    encoded = {
        name: image_base64(path, (520, 650))
        for name, path in EXTENSION_IMAGES.items()
    }
    cells = [
        intro(
            8,
            "실무 적용 시나리오 설계 및 최종 정리",
            "poc_candidate_card.md",
            "견적서·신청서·거래명세서 실물 사진을 비교하고 첫 PoC 한 가지를 고릅니다.",
        ),
        runtime_cell(),
        code(
            f"""
            import base64
            import io
            from PIL import Image
            try:
                from IPython.display import display
            except ImportError:
                display = lambda image: None

            EXTENSION_IMAGES = {encoded!r}
            EXTENSION_EXAMPLES = {EXTENSION_EXAMPLES!r}
            for key, payload in EXTENSION_IMAGES.items():
                image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
                image.thumbnail((320, 400))
                print(key, image.size)
                display(image)
            """
        ),
        markdown(
            """
            ## 형식이 바뀌면 생기는 어려움

            - **Excel**: 수식, 병합 셀, 숨김 시트, 숫자 서식
            - **Word**: 머리글, 텍스트박스, 변경 추적, 이미지로 삽입된 본문
            - **PDF**: 텍스트·스캔 혼합 페이지, 암호, 깨진 문자맵
            - **PPT**: 그룹 도형, 읽기 순서, 발표자 노트
            - **표 캡처**: 셀 관계가 사라져 행·열 위상을 다시 복원해야 함
            """
        ),
        code(
            """
            from textwrap import dedent

            candidate = "transaction_statement"
            example = EXTENSION_EXAMPLES[candidate]
            score = {
                "반복량": 4,
                "필드 안정성": 4,
                "오류 영향": 2,
                "예외 빈도": 3,
                "사람 검토 가능성": 5,
            }
            recommendation = (
                "GO_SMALL"
                if score["반복량"] >= 4 and score["사람 검토 가능성"] >= 4
                else "REVIEW"
            )
            card = f'''# 문서 자동화 PoC 후보 카드

            | 항목 | 내용 |
            | --- | --- |
            | 선택 문서 | {example["name"]} |
            | 추출 필드 | {", ".join(example["fields"])} |
            | 검증 규칙 | {" / ".join(example["rules"])} |
            | 틀렸을 때 영향 | {example["risk"]} |
            | 입력 제한 | 승인된 비식별 한 장 |
            | 최종 산출물 | 사람 승인 후 Excel |
            | 제안 | {recommendation} |

            ## 첫 PoC 통과 기준

            - 같은 양식 30장을 모아 정답표와 비교한다.
            - 필드별 정확도뿐 아니라 수정률과 처리시간을 기록한다.
            - 오류 시 자동 저장하지 않고 검토 대기열로 보낸다.
            - 개인정보·보존·삭제 정책을 먼저 승인받는다.
            '''
            output_path = OUTPUT_DIR / "poc_candidate_card.md"
            output_path.write_text(dedent(card), encoding="utf-8")
            print(dedent(card))
            print("CHECKPOINT 1/1 PASS:", output_path)
            """
        ),
    ]
    return notebook("08_business_application.ipynb", cells)


BUILDERS = {
    1: notebook_01,
    2: notebook_02,
    3: notebook_03,
    4: notebook_04,
    5: notebook_05,
    6: notebook_06,
    7: notebook_07,
    8: notebook_08,
}


def main() -> None:
    COLAB_DIR.mkdir(parents=True, exist_ok=True)
    for lesson, builder in BUILDERS.items():
        path = COLAB_DIR / f"{lesson:02d}_{NOTEBOOK_SLUGS[lesson]}.ipynb"
        path.write_text(
            json.dumps(builder(), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print("생성:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
