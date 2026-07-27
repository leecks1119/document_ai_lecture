"""독립 실행 가능한 1~8교시 Google Colab 노트북을 생성한다."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from textwrap import dedent

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"
SAMPLE_IMAGE = ROOT / "sample_docs" / "receipt_sample.png"
KOREAN_RECEIPT_IMAGE = (
    ROOT
    / "sample_docs"
    / "public_receipts"
    / "korea"
    / "taebaek_restaurant_2025_redacted.png"
)

SAMPLE_TEXT = """샘플문구점
거래일자: 2026-07-27
연필 2개 × 1,000원 = 2,000원
노트 1개 × 3,000원 = 3,000원
합계: 5,000원
"""

SAMPLE_VLM_MARKDOWN = """# 샘플문구점

거래일자: 2026-07-27

| 품목 | 수량 | 단가 | 금액 |
|---|---:|---:|---:|
| 연필 | 2 | 1,000원 | 2,000원 |
| 노트 | 1 | 3,000원 | 3,000원 |

**합계: 5,000원**
"""

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
            "colab": {
                "name": name,
                "provenance": [],
            },
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


def intro(
    lesson: int,
    title: str,
    artifact: str,
    goal: str,
) -> dict:
    return markdown(
        f"""
        # {lesson}교시. {title}

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/document_ai_lecture_2026/colab/{lesson:02d}_{NOTEBOOK_SLUGS[lesson]}.ipynb)

        **목표:** {goal}

        **결과물:** `{artifact}`

        - 기본 경로는 API 키와 OCR 모델 다운로드가 필요 없습니다.
        - 선택 실습은 기본값이 `False`입니다.
        - 수업이 지정한 공개 실물·합성 샘플만 사용합니다.
        """
    )


def runtime_cell() -> dict:
    return code(
        """
        import platform
        import sys
        from pathlib import Path

        OUTPUT_DIR = Path("course_outputs")
        OUTPUT_DIR.mkdir(exist_ok=True)

        print("Python:", sys.version.split()[0])
        print("Platform:", platform.platform())
        print("Output:", OUTPUT_DIR.resolve())
        """
    )


def json_literal(value) -> str:
    return repr(value)


def notebook_image_base64(path: Path, max_size: tuple[int, int]) -> str:
    """Colab에 넣을 메타데이터 없는 축소 JPEG를 만든다."""

    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(max_size)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def receipt_constants() -> str:
    return (
        f"SAMPLE_OCR_TEXT = {SAMPLE_TEXT!r}\n"
        f"SAMPLE_VLM_MARKDOWN = {SAMPLE_VLM_MARKDOWN!r}\n"
        f"SAMPLE_RECEIPT = {json_literal(SAMPLE_RECEIPT)}\n"
    )


def notebook_01() -> dict:
    encoded_image = notebook_image_base64(KOREAN_RECEIPT_IMAGE, (900, 1100))
    cells = [
        intro(
            1,
            "한국 영수증으로 구분하는 OCR·VLM·Document AI",
            "technology_comparison.json",
            "같은 한국 영수증에서 세 기술의 역할과 처리 과정을 구분합니다.",
        ),
        runtime_cell(),
        code(
            f"""
            import base64
            import io
            from PIL import Image

            KOREAN_RECEIPT_BASE64 = {encoded_image!r}
            receipt_image = Image.open(
                io.BytesIO(base64.b64decode(KOREAN_RECEIPT_BASE64))
            ).convert("RGB")
            receipt_image.thumbnail((720, 900))
            print("실제 한국 영수증 파생본:", receipt_image.size)
            receipt_image
            """
        ),
        markdown(
            """
            ## 핵심 3개

            1. OCR: 픽셀 → 영역 탐지 → 문자 인식 → 텍스트·좌표·신뢰도
            2. VLM: 이미지+지시 → 배치·언어 관계 해석 → 표·Markdown·초안 JSON
            3. Document AI: 처리기 선택 → 스키마 → 검증 → 사람 확인 → 저장

            아래 결과는 모두 **교육용 예시이며 실제 모델 실행 결과가 아닙니다.**
            """
        ),
        code(
            """
            COMPARISONS = [
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
            ]

            for item in COMPARISONS:
                print(item["technology"], "→", item["output"])
            """
        ),
        markdown("## 실습. 한 장의 영수증을 세 방식으로 비교"),
        code(
            """
            TECHNOLOGY_COMPARISON = {
                "input_document": "taebaek_restaurant_2025_redacted.png",
                "example_label": "교육용 예시 — 실제 모델 실행 결과가 아님",
                "comparisons": COMPARISONS,
                "document_ai_workflow": [
                    "입력 품질",
                    "OCR·VLM·혼합",
                    "업무 스키마",
                    "규칙 검증",
                    "사람 확인",
                    "저장",
                ],
            }

            technologies = {
                item["technology"] for item in TECHNOLOGY_COMPARISON["comparisons"]
            }
            assert technologies == {"OCR", "VLM", "Document AI"}
            TECHNOLOGY_COMPARISON
            """
        ),
        code(
            """
            import json

            output_path = OUTPUT_DIR / "technology_comparison.json"
            output_path.write_text(
                json.dumps(
                    TECHNOLOGY_COMPARISON,
                    ensure_ascii=False,
                    indent=2,
                ) + "\\n",
                encoding="utf-8",
            )
            print("저장 완료:", output_path)
            """
        ),
        markdown(
            """
            ## 확인

            - OCR·VLM·Document AI가 한 번씩 있는가?
            - 각 기술의 입력·과정·출력·한계가 구분되는가?
            - 교육용 예시 라벨이 남아 있는가?
            """
        ),
    ]
    return notebook("01_document_ai_overview.ipynb", cells)


def notebook_02() -> dict:
    encoded_image = base64.b64encode(SAMPLE_IMAGE.read_bytes()).decode("ascii")
    cells = [
        intro(
            2,
            "OCR 결과를 눈으로 확인하기",
            "ocr_text.txt",
            "OCR 결과의 텍스트·위치·신뢰도를 읽고 원본과 비교합니다.",
        ),
        runtime_cell(),
        code(
            f"""
            import base64
            import io
            from PIL import Image, ImageDraw

            SAMPLE_IMAGE_BASE64 = {encoded_image!r}
            receipt_image = Image.open(
                io.BytesIO(base64.b64decode(SAMPLE_IMAGE_BASE64))
            ).convert("RGB")
            MOCK_OCR_RESULT = {json_literal(SAMPLE_OCR_RESULT)}
            print("MOCK OCR 결과:", len(MOCK_OCR_RESULT), "개 영역")
            """
        ),
        markdown(
            """
            ## 핵심 3개

            1. OCR 결과는 텍스트·위치·신뢰도를 포함할 수 있습니다.
            2. 흐림과 기울기는 오류 가능성을 높입니다.
            3. 신뢰도는 정답이 아니라 검토 신호입니다.
            """
        ),
        code(
            """
            def draw_boxes(image, result):
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                for item in result:
                    xs = [point[0] for point in item["box"]]
                    ys = [point[1] for point in item["box"]]
                    draw.rectangle(
                        (min(xs), min(ys), max(xs), max(ys)),
                        outline="#167D7F",
                        width=4,
                    )
                return annotated

            annotated = draw_boxes(receipt_image, MOCK_OCR_RESULT)
            annotated_path = OUTPUT_DIR / "ocr_boxes.png"
            annotated.save(annotated_path)
            print("바운딩 박스 이미지:", annotated_path)
            """
        ),
        markdown("## 실습. 텍스트만 읽기 순서대로 저장"),
        code(
            """
            ocr_text = "\\n".join(item["text"] for item in MOCK_OCR_RESULT)
            output_path = OUTPUT_DIR / "ocr_text.txt"
            output_path.write_text(ocr_text + "\\n", encoding="utf-8")

            print(ocr_text)
            print("저장 완료:", output_path)
            """
        ),
        markdown(
            """
            ## 선택 실습. PaddleOCR 3.7 + PP-OCRv5 Korean

            한국어 인식은 `lang="korean"`과 `PP-OCRv5`를 사용합니다.
            최초 모델 다운로드가 가능할 때만 실행하며, 실패해도 위 결과물은 이미 완성됐습니다.
            """
        ),
        code(
            """
            RUN_PADDLEOCR = False

            if RUN_PADDLEOCR:
                import subprocess

                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "paddlepaddle==3.2.1",
                    ]
                )
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "paddleocr==3.7.0"]
                )
                from paddleocr import PaddleOCR

                image_path = OUTPUT_DIR / "receipt_sample.png"
                receipt_image.save(image_path)
                pipeline = PaddleOCR(
                    lang="korean",
                    ocr_version="PP-OCRv5",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                )
                live_pages = list(pipeline.predict(str(image_path)))
                live_payload = live_pages[0].json
                if callable(live_payload):
                    live_payload = live_payload()
                live_result = live_payload.get("res", live_payload)
                print("LIVE PaddleOCR 텍스트:", live_result.get("rec_texts", []))
            else:
                print("선택 PaddleOCR 셀을 건너뛰었습니다.")
            """
        ),
        markdown(
            """
            ## 확인

            - `ocr_text.txt`가 생성됐는가?
            - 높은 신뢰도도 원문과 비교했는가?
            - 선택 OCR 실패가 기본 실습을 막지 않는가?
            """
        ),
    ]
    return notebook("02_ocr_basic.ipynb", cells)


def notebook_03() -> dict:
    cells = [
        intro(
            3,
            "OCR 초안을 정돈된 데이터로 바꾸기",
            "clean_receipt.json",
            "원문·정제 결과·변경 기록을 함께 보존합니다.",
        ),
        runtime_cell(),
        code(
            f"""
            import json
            import re

            SAMPLE_OCR_TEXT = {SAMPLE_TEXT!r}
            """
        ),
        markdown(
            """
            ## 핵심 3개

            1. 키-값과 반복 품목은 다른 구조입니다.
            2. OCR 줄 순서가 논리적 순서와 다를 수 있습니다.
            3. 정제는 없는 값을 만드는 단계가 아닙니다.
            """
        ),
        code(
            """
            def normalize_line(line):
                original = line
                cleaned = re.sub(r"\\s+", " ", line.strip())
                changes = []
                if original != cleaned:
                    changes.append(f"공백 정리: {original!r} → {cleaned!r}")
                return cleaned, changes


            def group_receipt_lines(raw_text):
                cleaned_lines = []
                change_log = []
                for raw_line in raw_text.splitlines():
                    cleaned, changes = normalize_line(raw_line)
                    if cleaned:
                        cleaned_lines.append(cleaned)
                        change_log.extend(changes)

                groups = {"header": [], "date": [], "items": [], "total": [], "other": []}
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
            """
        ),
        markdown("## 실습. 원문을 보존하며 네 영역으로 분류"),
        code(
            """
            clean_result = group_receipt_lines(SAMPLE_OCR_TEXT)

            assert clean_result["raw_text"] == SAMPLE_OCR_TEXT
            assert len(clean_result["groups"]["items"]) == 2
            assert clean_result["groups"]["total"] == ["합계: 5,000원"]

            output_path = OUTPUT_DIR / "clean_receipt.json"
            output_path.write_text(
                json.dumps(clean_result, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            print(json.dumps(clean_result["groups"], ensure_ascii=False, indent=2))
            print("저장 완료:", output_path)
            """
        ),
        markdown(
            """
            ## mock 대체 경로

            외부 `ocr_text.txt`가 없어도 내장 `SAMPLE_OCR_TEXT`를 같은 함수에 넣습니다.
            정제 단계를 건너뛰지 않습니다.
            """
        ),
        markdown(
            """
            ## 확인

            - 원문 `raw_text`가 그대로 남았는가?
            - 품목이 두 줄인가?
            - 원문에 없던 값을 추가하지 않았는가?
            """
        ),
    ]
    return notebook("03_document_structure.ipynb", cells)


def notebook_04() -> dict:
    encoded_image = base64.b64encode(SAMPLE_IMAGE.read_bytes()).decode("ascii")
    cells = [
        intro(
            4,
            "멀티모달로 문서 구조 읽기",
            "receipt.json",
            "PaddleOCR-VL의 구조화 결과를 이해하고 업무용 JSON으로 변환합니다.",
        ),
        runtime_cell(),
        code(
            "import base64\n"
            "import io\n"
            "import json\n"
            "import re\n"
            "from PIL import Image\n\n"
            f"SAMPLE_IMAGE_BASE64 = {encoded_image!r}\n"
            + receipt_constants()
        ),
        markdown(
            """
            ## 핵심 3개

            1. VLM은 문서 이미지의 글자와 표·제목 같은 배치를 함께 봅니다.
            2. PaddleOCR-VL의 Markdown·블록 결과는 업무 JSON 전의 중간 결과입니다.
            3. 원문에 없는 값은 `null`로 두고 스키마와 업무 규칙으로 검증합니다.
            """
        ),
        code(
            """
            RECEIPT_SCHEMA = {
                "date": {"type": ["string", "null"]},
                "total_amount": {"type": ["integer", "null"], "minimum": 0},
            }


            def to_int(value):
                return int(value.replace(",", ""))


            def mock_extract(vlm_markdown):
                lines = [
                    line.strip()
                    for line in vlm_markdown.splitlines()
                    if line.strip()
                ]
                date_match = re.search(r"\\d{4}-\\d{2}-\\d{2}", vlm_markdown)
                total_match = re.search(
                    r"합계\\s*:\\s*([\\d,]+)원",
                    vlm_markdown.replace("*", ""),
                )
                item_pattern = re.compile(
                    r"(?P<name>.+?)\\s+(?P<quantity>\\d+)개\\s*[×x]\\s*"
                    r"(?P<unit>[\\d,]+)원\\s*=\\s*(?P<line>[\\d,]+)원"
                )
                items = []
                for line in lines:
                    match = item_pattern.search(line)
                    if match:
                        items.append({
                            "name": match.group("name").strip(),
                            "quantity": int(match.group("quantity")),
                            "unit_price": to_int(match.group("unit")),
                            "line_total": to_int(match.group("line")),
                        })
                        continue

                    if line.startswith("|") and "---" not in line:
                        cells = [cell.strip() for cell in line.strip("|").split("|")]
                        if len(cells) == 4 and cells[0] not in ("품목", ""):
                            try:
                                quantity = int(cells[1])
                                unit_price = to_int(cells[2].removesuffix("원"))
                                line_total = to_int(cells[3].removesuffix("원"))
                            except ValueError:
                                continue
                            items.append({
                                "name": cells[0],
                                "quantity": quantity,
                                "unit_price": unit_price,
                                "line_total": line_total,
                            })

                store_name = next(
                    (
                        line.removeprefix("#").strip()
                        for line in lines
                        if line.startswith("#")
                    ),
                    None,
                )
                return {
                    "document_type": "receipt",
                    "store_name": store_name,
                    "date": date_match.group(0) if date_match else None,
                    "total_amount": to_int(total_match.group(1)) if total_match else None,
                    "items": items,
                    "source_mode": "mock_vlm",
                }
            """
        ),
        markdown("## 실습. VLM 중간 결과를 업무용 JSON으로 변환"),
        code(
            """
            print("PaddleOCR-VL 형태의 Markdown 중간 결과:")
            print(SAMPLE_VLM_MARKDOWN)

            receipt = mock_extract(SAMPLE_VLM_MARKDOWN)

            assert receipt["date"] == "2026-07-27"
            assert receipt["total_amount"] == 5000
            assert len(receipt["items"]) == 2

            output_path = OUTPUT_DIR / "receipt.json"
            output_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            """
        ),
        markdown(
            """
            ## 선택 실습. PaddleOCR-VL 1.6

            문서 전용 멀티모달 모델은 다운로드와 더 많은 메모리가 필요합니다.
            Colab 런타임에서만 선택적으로 실행하며, 기본 mock 실습은 이미 완료됐습니다.
            """
        ),
        code(
            """
            RUN_PADDLEOCR_VL = False

            if RUN_PADDLEOCR_VL:
                import subprocess

                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "paddleocr[doc-parser]==3.7.0",
                        "transformers>=5.8,<6",
                    ]
                )
                from paddleocr import PaddleOCRVL

                image_path = OUTPUT_DIR / "receipt_sample.png"
                Image.open(
                    io.BytesIO(base64.b64decode(SAMPLE_IMAGE_BASE64))
                ).convert("RGB").save(image_path)

                pipeline = PaddleOCRVL(
                    pipeline_version="v1.6",
                    engine="transformers",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                )
                live_pages = list(pipeline.predict(str(image_path)))
                for page in live_pages:
                    print(page.markdown)
            else:
                print("선택 PaddleOCR-VL 셀을 건너뛰고 mock 결과로 완료했습니다.")
            """
        ),
        markdown(
            """
            ## 확인

            - 없는 값 규칙이 `null`인가?
            - 합계가 숫자 `5000`인가?
            - JSON 모양과 원문 근거를 따로 확인했는가?
            """
        ),
    ]
    return notebook("04_genai_extraction.ipynb", cells)


GRADIO_SETUP = """
import importlib.metadata
import subprocess

required_gradio = "6.20.0"
try:
    installed_gradio = importlib.metadata.version("gradio")
except importlib.metadata.PackageNotFoundError:
    installed_gradio = None

if installed_gradio != required_gradio:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", f"gradio=={required_gradio}"]
    )

import gradio as gr
print("Gradio:", gr.__version__)
"""


def notebook_05() -> dict:
    cells = [
        intro(
            5,
            "Python 함수에 화면 붙이기",
            "app_05.py",
            "Gradio 버튼과 mock 처리 함수를 연결합니다.",
        ),
        runtime_cell(),
        code(GRADIO_SETUP),
        code(receipt_constants()),
        markdown(
            """
            ## 핵심 3개

            1. 컴포넌트는 함수 입출력을 화면에 연결합니다.
            2. 버튼 이벤트에는 함수·입력·출력이 필요합니다.
            3. 화면과 처리 함수는 따로 확인합니다.
            """
        ),
        code(
            """
            def show_mock_result(file_path=None, processor="PaddleOCR"):
                status = f"MOCK {processor} 결과 — 업로드 문서를 읽지 않았습니다."
                source_text = (
                    SAMPLE_VLM_MARKDOWN
                    if processor == "PaddleOCR-VL"
                    else SAMPLE_OCR_TEXT
                )
                return status, source_text, SAMPLE_RECEIPT


            direct_result = show_mock_result()
            assert "MOCK PaddleOCR 결과" in direct_result[0]
            assert direct_result[2]["total_amount"] == 5000
            print(direct_result[0])
            """
        ),
        markdown("## 실습. 버튼과 함수 연결"),
        code(
            """
            with gr.Blocks(title="5교시 Document AI") as demo:
                gr.Markdown("# 영수증 Document AI · MOCK 실습")
                file_input = gr.File(label="합성 영수증", type="filepath")
                processor = gr.Radio(
                    ["PaddleOCR", "PaddleOCR-VL"],
                    value="PaddleOCR",
                    label="문서 처리기",
                )
                process_button = gr.Button("mock 결과 보기", variant="primary")
                status = gr.Markdown()
                ocr_output = gr.Textbox(label="문서 인식 결과", lines=9)
                json_output = gr.JSON(label="구조화 JSON")

                process_button.click(
                    fn=show_mock_result,
                    inputs=[file_input, processor],
                    outputs=[status, ocr_output, json_output],
                )

            print("Gradio 화면 구성 완료")
            """
        ),
        code(
            """
            app_code = '''import gradio as gr

            SAMPLE_TEXT = "합성 영수증 OCR 텍스트"
            SAMPLE_JSON = {"source_mode": "mock"}

            def show_mock_result(file_path=None, processor="PaddleOCR"):
                return f"MOCK {processor} 결과", SAMPLE_TEXT, SAMPLE_JSON

            with gr.Blocks() as demo:
                file_input = gr.File(type="filepath")
                processor = gr.Radio(["PaddleOCR", "PaddleOCR-VL"], value="PaddleOCR")
                button = gr.Button("mock 결과 보기")
                status = gr.Markdown()
                text = gr.Textbox()
                data = gr.JSON()
                button.click(
                    show_mock_result,
                    [file_input, processor],
                    [status, text, data],
                )

            if __name__ == "__main__":
                demo.launch(share=False)
            '''
            output_path = OUTPUT_DIR / "app_05.py"
            output_path.write_text(app_code, encoding="utf-8")
            print("저장 완료:", output_path)
            """
        ),
        code(
            """
            RUN_PUBLIC_DEMO = False

            if RUN_PUBLIC_DEMO:
                demo.launch(share=True, max_file_size="5mb")
            else:
                print("공개 공유 주소를 만들지 않았습니다. demo 객체 생성으로 검증했습니다.")
            """
        ),
        markdown(
            """
            ## mock 대체 경로

            Gradio 화면이 열리지 않아도 `show_mock_result()` 직접 실행 결과가 같으면 완료입니다.
            """
        ),
    ]
    return notebook("05_gradio_basic.ipynb", cells)


def notebook_06() -> dict:
    cells = [
        intro(
            6,
            "작은 함수들을 한 줄로 연결하기",
            "app_06.py",
            "오류를 숨기지 않고 실제·mock 경로를 연결합니다.",
        ),
        runtime_cell(),
        code(receipt_constants()),
        markdown(
            """
            ## 핵심 3개

            1. 단계를 작은 함수로 나눕니다.
            2. 오류와 사용 모드를 화면에 표시합니다.
            3. mock은 사용자가 명시적으로 선택합니다.
            """
        ),
        code(
            """
            def validate_upload(file_path):
                if not file_path:
                    return ["파일을 선택하세요."]
                return []


            def mock_extract(ocr_text):
                data = dict(SAMPLE_RECEIPT)
                data["source_mode"] = "mock_extraction"
                return data


            def process_document(file_path=None, *, processor="ocr", use_sample=False):
                if processor not in ("ocr", "vlm"):
                    return {
                        "ok": False,
                        "status": "입력 오류",
                        "errors": ["processor는 ocr 또는 vlm이어야 합니다."],
                    }

                if use_sample:
                    document_text = (
                        SAMPLE_VLM_MARKDOWN if processor == "vlm" else SAMPLE_OCR_TEXT
                    )
                    status = (
                        "MOCK PaddleOCR-VL + MOCK 추출"
                        if processor == "vlm"
                        else "MOCK PaddleOCR + MOCK 추출"
                    )
                else:
                    errors = validate_upload(file_path)
                    if errors:
                        return {
                            "ok": False,
                            "status": "입력 오류",
                            "errors": errors,
                            "can_continue_with_sample": True,
                        }
                    return {
                        "ok": False,
                        "status": "실제 모델 선택 실행 필요",
                        "errors": [
                            "PaddleOCR 또는 PaddleOCR-VL 선택 실습을 실행하거나 "
                            "'샘플로 계속'을 선택하세요."
                        ],
                        "can_continue_with_sample": True,
                    }

                return {
                    "ok": True,
                    "status": status,
                    "document_text": document_text,
                    "data": mock_extract(document_text),
                }
            """
        ),
        markdown("## 실습. 오류 경로와 명시적 mock 경로 확인"),
        code(
            """
            error_result = process_document()
            assert not error_result["ok"]
            assert "data" not in error_result

            sample_result = process_document(processor="vlm", use_sample=True)
            assert sample_result["ok"]
            assert "MOCK" in sample_result["status"]
            assert sample_result["data"]["total_amount"] == 5000

            print(error_result["status"], "→ 사용자가 샘플 선택")
            print(sample_result["status"], "→ 완료")
            """
        ),
        code(
            """
            app_code = '''from src.pipeline import process_document

            def run_uploaded(file_path, processor="ocr"):
                return process_document(file_path, processor=processor)

            def run_sample(processor="ocr"):
                return process_document(processor=processor, use_sample=True)

            # Gradio에서는 두 함수를 서로 다른 버튼에 연결합니다.
            '''
            output_path = OUTPUT_DIR / "app_06.py"
            output_path.write_text(app_code, encoding="utf-8")
            print("저장 완료:", output_path)
            """
        ),
        markdown(
            """
            ## mock 대체 경로

            필수 경로 자체가 명시적인 mock 실습입니다. 실제 모델 오류 뒤에 자동 전환하지 않고
            오류를 확인한 뒤 처리기를 골라 `process_document(use_sample=True)`를 실행합니다.
            """
        ),
        markdown(
            """
            ## 확인

            - 오류 결과에 관련 없는 JSON이 없는가?
            - mock 상태가 화면에 분명히 표시되는가?
            - 사용자가 직접 샘플 경로를 선택했는가?
            """
        ),
    ]
    return notebook("06_ocr_ai_integration.ipynb", cells)


def notebook_07() -> dict:
    cells = [
        intro(
            7,
            "틀린 값을 걸러 CSV로 저장하기",
            "receipt.csv",
            "필수값과 품목 합계를 확인하고 CSV를 만듭니다.",
        ),
        runtime_cell(),
        code(
            f"""
            import csv
            import io
            from copy import deepcopy

            SAMPLE_RECEIPT = {json_literal(SAMPLE_RECEIPT)}
            MISSING_STORE = deepcopy(SAMPLE_RECEIPT)
            MISSING_STORE["store_name"] = None
            WRONG_TOTAL = deepcopy(SAMPLE_RECEIPT)
            WRONG_TOTAL["total_amount"] = 6000
            """
        ),
        markdown(
            """
            ## 핵심 3개

            1. 검증 결과는 valid·warnings·errors로 나눕니다.
            2. 자료형과 업무 규칙은 다른 검사입니다.
            3. 품목 하나가 CSV의 한 행이 됩니다.
            """
        ),
        code(
            """
            def validate_receipt(data):
                errors = []

                for field in ("store_name", "date", "total_amount", "items"):
                    if data.get(field) in (None, "", []):
                        errors.append(f"필수값 누락: {field}")

                item_sum = sum(
                    item["line_total"] for item in data.get("items", [])
                )
                if data.get("total_amount") != item_sum:
                    errors.append("품목 합계와 총액이 다릅니다.")

                return {
                    "valid": not errors,
                    "warnings": [],
                    "errors": errors,
                }
            """
        ),
        markdown("## 실습. 세 데이터 검증"),
        code(
            """
            normal = validate_receipt(SAMPLE_RECEIPT)
            missing = validate_receipt(MISSING_STORE)
            wrong_total = validate_receipt(WRONG_TOTAL)

            assert normal["valid"]
            assert not missing["valid"]
            assert not wrong_total["valid"]

            print("정상:", normal)
            print("누락:", missing)
            print("합계 불일치:", wrong_total)
            """
        ),
        code(
            """
            def safe_text(value):
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
                    return "'" + value
                return value


            def receipt_rows(data):
                rows = []
                for item in data["items"]:
                    row = {
                        "store_name": data["store_name"],
                        "date": data["date"],
                        "total_amount": data["total_amount"],
                        "item_name": item["name"],
                        "quantity": item["quantity"],
                        "unit_price": item["unit_price"],
                        "line_total": item["line_total"],
                    }
                    rows.append({key: safe_text(value) for key, value in row.items()})
                return rows


            columns = [
                "store_name", "date", "total_amount", "item_name",
                "quantity", "unit_price", "line_total",
            ]
            output_path = OUTPUT_DIR / "receipt.csv"
            with output_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(receipt_rows(SAMPLE_RECEIPT))

            print("저장 완료:", output_path)
            print(output_path.read_text(encoding="utf-8-sig"))
            """
        ),
        code(
            """
            RUN_COLAB_DOWNLOAD = False

            if RUN_COLAB_DOWNLOAD:
                from google.colab import files
                files.download(str(output_path))
            else:
                print("자동 다운로드를 건너뛰었습니다. Colab 파일 영역에서 받을 수 있습니다.")
            """
        ),
        markdown(
            """
            ## mock 대체 경로

            이전 앱 없이 내장 `SAMPLE_RECEIPT`를 같은 검증·CSV 함수에 전달합니다.
            """
        ),
    ]
    return notebook("07_validation_export.ipynb", cells)


def notebook_08() -> dict:
    cells = [
        intro(
            8,
            "자동화의 마지막 단계는 사람의 확인",
            "business_application_card.md",
            "전체 흐름을 점검하고 사람 검토가 있는 적용 카드를 씁니다.",
        ),
        runtime_cell(),
        code(
            f"""
            from copy import deepcopy

            SAMPLE_RECEIPT = {json_literal(SAMPLE_RECEIPT)}

            def validate_receipt(data):
                errors = []
                for field in ("store_name", "date", "total_amount", "items"):
                    if data.get(field) in (None, "", []):
                        errors.append(f"필수값 누락: {{field}}")
                item_sum = sum(
                    item["line_total"] for item in data.get("items", [])
                )
                if data.get("total_amount") != item_sum:
                    errors.append("품목 합계와 총액이 다릅니다.")
                return {{"valid": not errors, "warnings": [], "errors": errors}}

            def run_smoke_test():
                missing = deepcopy(SAMPLE_RECEIPT)
                missing["store_name"] = None
                wrong_total = deepcopy(SAMPLE_RECEIPT)
                wrong_total["total_amount"] = 6000
                return {{
                    "mock_path_works": SAMPLE_RECEIPT["source_mode"] == "mock",
                    "normal_result_is_valid": validate_receipt(SAMPLE_RECEIPT)["valid"],
                    "missing_required_is_blocked": not validate_receipt(missing)["valid"],
                    "wrong_total_is_blocked": not validate_receipt(wrong_total)["valid"],
                }}
            """
        ),
        markdown(
            """
            ## 핵심 3개

            1. 자동화 후보는 반복량·오류 영향·예외 빈도로 봅니다.
            2. 개인정보·외부 전송·보존·삭제를 확인합니다.
            3. 사람의 최종 승인과 수정 절차를 정합니다.
            """
        ),
        code(
            """
            smoke_result = run_smoke_test()
            assert all(smoke_result.values())
            print(smoke_result)
            """
        ),
        markdown("## 실습. 한 장짜리 적용 카드"),
        code(
            """
            card = '''# 문서 자동화 업무 적용 카드

            | 항목 | 작성 내용 |
            | --- | --- |
            | 입력 문서 | 사내 비용 처리용 영수증 |
            | 필요한 추출 필드 | 상호명, 날짜, 품목, 총액 |
            | 틀렸을 때의 영향 | 총액 오류 시 정산 금액이 달라짐 |
            | 사람 검토자 | 비용 처리 담당자 |
            | 저장 형식과 위치 | 승인 후 CSV, 승인된 저장소 |
            | 원본·결과 삭제 시점 | 조직의 보존 기준에 따름 |

            ## 적용 전 확인

            - [x] 합성 문서로 기능을 점검했다.
            - [ ] 실제 개인정보의 외부 전송 승인을 확인한다.
            - [ ] 최종 승인자와 반려 절차를 확인한다.
            '''

            output_path = OUTPUT_DIR / "business_application_card.md"
            output_path.write_text(card, encoding="utf-8")
            print(card)
            print("저장 완료:", output_path)
            """
        ),
        markdown(
            """
            ## mock 대체 경로

            앱이 열리지 않아도 `run_smoke_test()` 결과와 제공 카드 템플릿으로 실습을 완료합니다.

            ## 최종 확인

            - 처리할 문서가 한 종류인가?
            - 오류 영향과 사람 검토자가 적혀 있는가?
            - 저장 위치와 삭제 시점이 적혀 있는가?
            """
        ),
    ]
    return notebook("08_business_application.ipynb", cells)


NOTEBOOK_SLUGS = {
    1: "document_ai_overview",
    2: "ocr_basic",
    3: "document_structure",
    4: "genai_extraction",
    5: "gradio_basic",
    6: "ocr_ai_integration",
    7: "validation_export",
    8: "business_application",
}


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
