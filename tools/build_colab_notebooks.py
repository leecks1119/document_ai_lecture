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

        - 모든 필수 실습은 Google Colab에서 진행합니다.
        - 학습자 API 키나 결제가 필요 없습니다.
        - 실행이 막히면 교재에 포함된 완성 복구본으로 같은 실습을 계속합니다.
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
            "receipt_pipeline_trace.json",
            "용어 관계를 구분하고 영수증 한 장의 0~12 전체 처리 흐름을 추적합니다.",
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
            ## 강사가 먼저 보여 줄 최종 목적지

            완성 `receipt_result.xlsx`에는 원문·최종값·검토 상태와 품목 행이 남습니다.
            1교시에는 파일을 직접 만들지 않고 목적지만 확인합니다. 실제 생성은 7교시입니다.
            """
        ),
        markdown(
            """
            ## 이 과정의 용어 기준

            업계에서 용어 경계는 공급자에 따라 일부 겹칩니다. 이 과정에서는 다음처럼 구분합니다.

            1. **OCR**: 이미지에서 텍스트를 인식합니다. 제품에 따라 페이지·좌표·신뢰도를 함께 제공할 수 있습니다.
            2. **Multimodal AI**: 둘 이상의 데이터 형식을 다루는 상위 범주입니다.
            3. **VLM**: 이미지와 언어를 함께 다루는 Multimodal AI의 한 종류입니다.
            4. **Document AI**: 분류·OCR·레이아웃·필드 추출·정규화 등 문서 구조화 역량입니다.
            5. **IDP**: Document AI 역량을 접수·검증·예외·사람 확인·업무 연결·운영 개선에 결합한 범위입니다.

            `OCR → VLM → Document AI`는 반드시 거치는 고정 순서가 아닙니다.
            """
        ),
        code(
            """
            TERM_RELATION = {
                "multimodal_ai": ["VLM"],
                "document_ai_capabilities": [
                    "OCR", "VLM", "분류", "레이아웃", "필드·표 추출", "정규화"
                ],
                "idp_operations": [
                    "접수", "보안", "검증", "예외 처리", "사람 확인",
                    "업무 연결", "관측·평가·개선",
                ],
            }

            assert "VLM" in TERM_RELATION["multimodal_ai"]
            assert "OCR" in TERM_RELATION["document_ai_capabilities"]
            TERM_RELATION
            """
        ),
        markdown(
            """
            ## 0~12 전체 참조 지도

            아래 단계는 암기 목록이 아닙니다. 하루 동안 각 실습이 어디에 있는지 찾는 지도입니다.

            | 번호 | 단계 | 영수증에서 확인할 질문 |
            |---:|---|---|
            | 0 | 목표·스키마 | 어떤 값이 필요하고 틀리면 어떤 문제가 생기는가? |
            | 1 | 접수 | 사진·PDF·Office 원본을 어디서 받는가? |
            | 2 | 형식 라우팅·분리 | 텍스트층·셀·픽셀 중 무엇인가? |
            | 3 | 품질 확인·전처리 | 흐림·잘림·회전이 처리 가능한가? |
            | 4 | 텍스트·레이아웃 추출 | 원본 파서인가, OCR인가? |
            | 5 | 문서 유형 분류 | 영수증인가, 다른 문서인가? |
            | 6 | 필드·표 구조화 | 상호명·날짜·품목·합계 후보는? |
            | 7 | 정규화 | `"76,000원"`을 `76000`으로 바꿔도 원문이 남는가? |
            | 8 | 검증 | 필수값·형식·계산 규칙이 맞는가? |
            | 9 | 처리 결정 | 자동 확정·사람 검토·처리 불가 중 무엇인가? |
            | 10 | 사람 검토 | 원본을 보고 승인·수정·반려했는가? |
            | 11 | 내보내기·연결 | 승인된 값을 어디로 보낼 것인가? |
            | 12 | 관측·평가·개선 | 어디서 자주 틀리고 비용이 드는가? |

            보안·접근 통제·감사 기록·보존·삭제는 모든 단계를 가로지릅니다.
            """
        ),
        code(
            """
            PIPELINE_MAP = [
                {"step": 0, "name": "목표·스키마", "output": "필드·검증·성공 기준"},
                {"step": 1, "name": "접수", "output": "원본·문서 ID"},
                {"step": 2, "name": "형식 라우팅·분리", "output": "처리 경로"},
                {"step": 3, "name": "품질 확인·전처리", "output": "처리 가능 입력"},
                {"step": 4, "name": "텍스트·레이아웃 추출", "output": "원문·위치"},
                {"step": 5, "name": "문서 유형 분류", "output": "문서 유형"},
                {"step": 6, "name": "필드·표 구조화", "output": "스키마 초안"},
                {"step": 7, "name": "정규화", "output": "원문·정규화 값"},
                {"step": 8, "name": "검증", "output": "규칙 결과"},
                {"step": 9, "name": "처리 결정", "output": "AUTO_ACCEPT·REVIEW·REJECT"},
                {"step": 10, "name": "사람 검토", "output": "승인·수정·반려 기록"},
                {"step": 11, "name": "내보내기·연결", "output": "업무 데이터"},
                {"step": 12, "name": "관측·평가·개선", "output": "품질·비용·개선안"},
            ]

            assert [item["step"] for item in PIPELINE_MAP] == list(range(13))
            print("전체 지도:", " → ".join(item["name"] for item in PIPELINE_MAP))
            """
        ),
        markdown("## 실습. 영수증 한 장의 처리 흔적 완성"),
        code(
            """
            PREPARED_TRACE = {
                "source_document": "taebaek_restaurant_2025_redacted.png",
                "source_mode": "교재 제작자가 원본에서 확인한 교육용 준비 결과",
                "schema_fields": ["net_amount", "tax_amount", "total_amount"],
                "raw_text": "공급가액 69,094 / 부가세 6,906 / 합계 76,000",
                "fields": {
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
                },
            }

            fields = PREPARED_TRACE["fields"]
            PREPARED_TRACE["validation"] = {
                "amount_math_ok": (
                    fields["net_amount"]["normalized_value"]
                    + fields["tax_amount"]["normalized_value"]
                    == fields["total_amount"]["normalized_value"]
                ),
                "evidence_present": all(
                    field["evidence"] for field in fields.values()
                ),
            }
            PREPARED_TRACE["routing_decision"] = "REVIEW"
            PREPARED_TRACE["routing_reason"] = "정책상 원본을 보는 사람 확인 필수"
            PREPARED_TRACE["human_decision"] = "APPROVED_AFTER_SOURCE_CHECK"
            PREPARED_TRACE["next_step"] = "7교시에서 receipt_result.xlsx 생성"

            assert PREPARED_TRACE["validation"]["amount_math_ok"]
            assert PREPARED_TRACE["routing_decision"] == "REVIEW"
            PREPARED_TRACE
            """
        ),
        markdown(
            """
            ## 처리 결정 비교

            - `AUTO_ACCEPT`: 필수값·형식·합계·근거 규칙을 모두 통과하고 오류 영향이 낮음
            - `REVIEW`: 값은 있으나 모호하거나 정책상 사람 확인이 필요함
            - `REJECT`: 입력 품질 부족, 필수 근거 없음, 지원하지 않는 문서

            합계 검증이 맞아도 사람 확인 정책 때문에 `REVIEW`가 될 수 있습니다.
            사람 검토는 원본에 없는 근거를 새로 만드는 단계가 아닙니다.
            """
        ),
        code(
            """
            import json

            output_path = OUTPUT_DIR / "receipt_pipeline_trace.json"
            output_path.write_text(
                json.dumps(
                    PREPARED_TRACE,
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

            - 원문·정규화 값·근거가 모두 남아 있는가?
            - 검증 결과와 처리 결정이 분리돼 있는가?
            - `REVIEW` 뒤 사람의 결정과 다음 단계가 기록됐는가?
            - 교육용 준비 결과 라벨이 남아 있는가?
            """
        ),
    ]
    return notebook("01_document_ai_overview.ipynb", cells)


def notebook_02() -> dict:
    encoded_image = base64.b64encode(SAMPLE_IMAGE.read_bytes()).decode("ascii")
    cells = [
        intro(
            2,
            "OCR 기반 텍스트 추출 실습",
            "ocr_result.json",
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
            OCR_RESULT = MOCK_OCR_RESULT
            SOURCE_MODE = "prepared"
            print("준비된 OCR 결과:", len(OCR_RESULT), "개 영역")
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

            annotated = draw_boxes(receipt_image, OCR_RESULT)
            annotated_path = OUTPUT_DIR / "ocr_boxes.png"
            annotated.save(annotated_path)
            print("바운딩 박스 이미지:", annotated_path)
            """
        ),
        markdown(
            """
            ## 기본 실습. PaddleOCR 3.7 + PP-OCRv5 Korean

            한국어 인식은 `lang="korean"`과 `PP-OCRv5`를 사용합니다.
            강사 안내에 따라 아래 값을 `True`로 바꿉니다. 설치·다운로드·실행이
            3분 안에 끝나지 않으면 중지하고 준비 결과로 같은 원본 대조 실습을 계속합니다.
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
                texts = live_result.get("rec_texts", [])
                boxes = live_result.get("rec_polys", [])
                scores = live_result.get("rec_scores", [])
                OCR_RESULT = [
                    {
                        "box": box,
                        "text": text,
                        "confidence": float(score),
                    }
                    for box, text, score in zip(boxes, texts, scores)
                ]
                SOURCE_MODE = "live_paddleocr"
                print("LIVE PaddleOCR 텍스트:", texts)
            else:
                print("준비된 OCR 결과로 원본 대조 실습을 계속합니다.")
            """
        ),
        markdown("## 실습. 원본 대조 표시와 함께 저장"),
        code(
            """
            import json

            reviewed = [
                {
                    **item,
                    "matches_source": None,
                    "review_note": "",
                }
                for item in OCR_RESULT
            ]
            output = {
                "source_mode": SOURCE_MODE,
                "items": reviewed,
            }
            output_path = OUTPUT_DIR / "ocr_result.json"
            output_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\\n",
                encoding="utf-8",
            )
            print("저장 완료:", output_path)
            """
        ),
        markdown(
            """
            ## 확인

            - `ocr_result.json`이 생성됐는가?
            - 높은 신뢰도도 원문과 비교했는가?
            - 실제 OCR이 3분 안에 실행되지 않을 때 준비 결과로 전환했는가?
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
            "멀티모달·생성형 AI 기반 핵심 정보 추출",
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
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"streamlit=={required_streamlit}",
        ]
    )

import streamlit
print("Streamlit:", streamlit.__version__)
"""


def notebook_05() -> dict:
    cells = [
        intro(
            5,
            "문서 자동화 웹 애플리케이션 기본 구현",
            "app_05.py",
            "Streamlit 파일 입력·버튼·결과 화면을 Python 처리 함수와 연결합니다.",
        ),
        runtime_cell(),
        code(STREAMLIT_SETUP),
        code(receipt_constants()),
        markdown(
            """
            ## 핵심 3개

            1. Streamlit은 위에서 아래로 실행되는 Python 스크립트를 웹앱으로 보여 줍니다.
            2. 파일 입력·실행 버튼·결과 영역을 처리 함수와 연결합니다.
            3. Colab에서는 브라우저 서버를 열지 않고 AppTest로 화면 코드를 검증합니다.
            """
        ),
        code(
            """
            from textwrap import dedent

            app_code = dedent(
                '''
                import streamlit as st

                SAMPLE_TEXT = "준비된 영수증 판독 결과"
                SAMPLE_JSON = {
                    "store_name": "샘플문구점",
                    "date": "2026-07-27",
                    "items": [],
                    "total_amount": 5000,
                    "source_mode": "prepared",
                }

                st.title("영수증 Document AI 미니 앱")
                uploaded = st.file_uploader(
                    "영수증 이미지 또는 PDF 한 장",
                    type=["png", "jpg", "jpeg", "pdf"],
                )
                if st.button("준비 결과로 실행"):
                    st.info("준비 결과를 사용했습니다.")
                    st.text_area("판독 원문", SAMPLE_TEXT)
                    st.json(SAMPLE_JSON)
                '''
            ).lstrip()
            output_path = OUTPUT_DIR / "app_05.py"
            output_path.write_text(app_code, encoding="utf-8")
            print("저장 완료:", output_path)
            """
        ),
        markdown("## 실습. Colab에서 Streamlit 앱 검사"),
        code(
            """
            from streamlit.testing.v1 import AppTest

            app_test = AppTest.from_file(str(output_path)).run(timeout=20)
            assert not app_test.exception
            assert app_test.title[0].value == "영수증 Document AI 미니 앱"
            assert len(app_test.file_uploader) == 1
            assert len(app_test.button) == 1
            print("Streamlit 화면 코드 검사 완료")
            """
        ),
        markdown(
            """
            ## 완성 복구본

            빈칸 수정이 어려우면 완성된 `app_05.py`를 저장하고 AppTest 결과를 확인합니다.
            실제 브라우저 서버나 공개 터널을 열지 않아도 이번 교시를 완료할 수 있습니다.
            """
        ),
    ]
    return notebook("05_streamlit_basic.ipynb", cells)


def notebook_06() -> dict:
    app_source = (
        dedent(
            f"""
            import streamlit as st

            SAMPLE_OCR_TEXT = {SAMPLE_TEXT!r}
            SAMPLE_VLM_MARKDOWN = {SAMPLE_VLM_MARKDOWN!r}
            SAMPLE_RECEIPT = {json_literal(SAMPLE_RECEIPT)}


            def process_document(uploaded=None, *, processor="ocr", use_sample=False):
                if processor not in ("ocr", "vlm"):
                    return {{
                        "ok": False,
                        "status": "입력 오류",
                        "errors": ["처리기는 ocr 또는 vlm이어야 합니다."],
                    }}

                if use_sample:
                    document_text = (
                        SAMPLE_VLM_MARKDOWN
                        if processor == "vlm"
                        else SAMPLE_OCR_TEXT
                    )
                    data = dict(SAMPLE_RECEIPT)
                    data["source_mode"] = f"MOCK {{processor}} prepared result"
                    return {{
                        "ok": True,
                        "status": f"MOCK {{processor.upper()}} + 준비 결과",
                        "document_text": document_text,
                        "data": data,
                    }}

                if uploaded is None:
                    return {{
                        "ok": False,
                        "status": "입력 오류",
                        "errors": ["파일을 선택하세요."],
                    }}

                return {{
                    "ok": False,
                    "status": "실제 모델 실행 필요",
                    "errors": [
                        "2교시 OCR 또는 4교시 VLM 선택 셀의 결과를 연결하세요."
                    ],
                }}


            st.title("영수증 Document AI 연결 앱")
            uploaded = st.file_uploader(
                "영수증 이미지 또는 PDF 한 장",
                type=["png", "jpg", "jpeg", "pdf"],
            )
            processor = st.radio(
                "처리기",
                options=["ocr", "vlm"],
                horizontal=True,
            )
            left, right = st.columns(2)
            run_uploaded = left.button("업로드 처리", key="run_uploaded")
            run_sample = right.button("샘플로 계속", key="run_sample")

            if run_uploaded:
                st.session_state["result"] = process_document(
                    uploaded,
                    processor=processor,
                )
            elif run_sample:
                st.session_state["result"] = process_document(
                    processor=processor,
                    use_sample=True,
                )

            result = st.session_state.get("result")
            if result:
                if result["ok"]:
                    st.success(result["status"])
                    st.text_area("판독 원문", result["document_text"])
                    st.json(result["data"])
                else:
                    st.error(result["status"])
                    for message in result["errors"]:
                        st.write(f"- {{message}}")
            """
        ).strip()
        + "\n"
    )
    cells = [
        intro(
            6,
            "OCR 및 정보 추출 기능 연동",
            "app_06.py",
            "오류를 숨기지 않고 실제·mock 경로를 연결합니다.",
        ),
        runtime_cell(),
        code(STREAMLIT_SETUP),
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
            f"""
            app_code = {app_source!r}
            output_path = OUTPUT_DIR / "app_06.py"
            output_path.write_text(app_code, encoding="utf-8")
            print("저장 완료:", output_path)
            """
        ),
        code(
            """
            from streamlit.testing.v1 import AppTest

            app_test = AppTest.from_file(str(output_path)).run(timeout=20)
            assert not app_test.exception
            assert app_test.title[0].value == "영수증 Document AI 연결 앱"
            assert len(app_test.file_uploader) == 1
            assert len(app_test.button) == 2

            app_test.button(key="run_sample").click().run(timeout=20)
            assert app_test.success
            assert "MOCK" in app_test.success[0].value
            print("독립 실행 Streamlit 앱 검사 완료")
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
            "추출 결과 검증 및 데이터 저장",
            "receipt_result.xlsx",
            "필수값과 품목 합계를 확인하고 검증된 결과만 Excel로 저장합니다.",
        ),
        runtime_cell(),
        code(
            f"""
            from copy import deepcopy
            from datetime import date
            from openpyxl import Workbook, load_workbook

            SAMPLE_RECEIPT = {json_literal(SAMPLE_RECEIPT)}
            SAMPLE_OCR_TEXT = {SAMPLE_TEXT!r}
            MISSING_STORE = deepcopy(SAMPLE_RECEIPT)
            MISSING_STORE["store_name"] = None
            WRONG_TOTAL = deepcopy(SAMPLE_RECEIPT)
            WRONG_TOTAL["total_amount"] = 6000
            BAD_DATE = deepcopy(SAMPLE_RECEIPT)
            BAD_DATE["date"] = "2026/07/27"
            BAD_AMOUNT = deepcopy(SAMPLE_RECEIPT)
            BAD_AMOUNT["total_amount"] = "5,000원"
            """
        ),
        markdown(
            """
            ## 핵심 3개

            1. 검증 결과는 valid·warnings·errors로 나눕니다.
            2. 자료형과 업무 규칙은 다른 검사입니다.
            3. 오류가 없고 사람이 확인한 결과만 Excel로 저장합니다.
            """
        ),
        code(
            """
            def is_iso_date(value):
                if not isinstance(value, str):
                    return False
                try:
                    date.fromisoformat(value)
                    return True
                except ValueError:
                    return False


            def validate_receipt(data):
                errors = []

                for field in ("store_name", "date", "total_amount", "items"):
                    if data.get(field) in (None, "", []):
                        errors.append(f"필수값 누락: {field}")

                if data.get("date") and not is_iso_date(data["date"]):
                    errors.append("date는 YYYY-MM-DD 형식이어야 합니다.")

                total_amount = data.get("total_amount")
                if total_amount is not None and (
                    not isinstance(total_amount, int) or total_amount < 0
                ):
                    errors.append("total_amount는 0 이상의 정수여야 합니다.")

                item_sum = sum(
                    item.get("line_total", 0)
                    for item in data.get("items", [])
                    if isinstance(item.get("line_total"), int)
                )
                if isinstance(total_amount, int) and total_amount != item_sum:
                    errors.append("품목 합계와 총액이 다릅니다.")

                return {
                    "valid": not errors,
                    "warnings": [],
                    "errors": errors,
                }
            """
        ),
        markdown("## 실습. 다섯 데이터 검증"),
        code(
            """
            normal = validate_receipt(SAMPLE_RECEIPT)
            missing = validate_receipt(MISSING_STORE)
            wrong_total = validate_receipt(WRONG_TOTAL)
            bad_date = validate_receipt(BAD_DATE)
            bad_amount = validate_receipt(BAD_AMOUNT)

            assert normal["valid"]
            assert not missing["valid"]
            assert not wrong_total["valid"]
            assert not bad_date["valid"]
            assert not bad_amount["valid"]

            print("정상:", normal)
            print("누락:", missing)
            print("합계 불일치:", wrong_total)
            print("날짜 형식:", bad_date)
            print("금액 형식:", bad_amount)
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


            def save_reviewed_excel(
                data,
                validation,
                *,
                human_approved,
                output_path,
                source_text,
            ):
                if not validation["valid"] or not human_approved:
                    return False

                raw_values = {
                    "store_name": data["store_name"],
                    "date": data["date"],
                    "total_amount": f'{data["total_amount"]:,}원',
                }
                workbook = Workbook()
                summary = workbook.active
                summary.title = "검토_요약"
                summary.append(
                    [
                        "field",
                        "raw_value",
                        "cleaned_value",
                        "final_value",
                        "review_status",
                    ]
                )
                for field in ("store_name", "date", "total_amount"):
                    summary.append(
                        [
                            field,
                            safe_text(raw_values[field]),
                            safe_text(data[field]),
                            safe_text(data[field]),
                            "사람 확인 완료",
                        ]
                    )

                items = workbook.create_sheet("품목")
                columns = [
                    "store_name", "date", "total_amount", "item_name",
                    "quantity", "unit_price", "line_total",
                ]
                items.append(columns)
                for row in receipt_rows(data):
                    items.append([row[column] for column in columns])

                source = workbook.create_sheet("원문")
                source.append(["source_mode", data["source_mode"]])
                source.append(["ocr_text", safe_text(source_text)])
                workbook.save(output_path)
                return True


            blocked_path = OUTPUT_DIR / "blocked_result.xlsx"
            assert not save_reviewed_excel(
                WRONG_TOTAL,
                wrong_total,
                human_approved=True,
                output_path=blocked_path,
                source_text=SAMPLE_OCR_TEXT,
            )
            assert not blocked_path.exists()

            not_approved_path = OUTPUT_DIR / "not_approved.xlsx"
            assert not save_reviewed_excel(
                SAMPLE_RECEIPT,
                normal,
                human_approved=False,
                output_path=not_approved_path,
                source_text=SAMPLE_OCR_TEXT,
            )
            assert not not_approved_path.exists()

            HUMAN_APPROVED = True  # 원본과 추출값을 직접 대조한 뒤에만 True
            output_path = OUTPUT_DIR / "receipt_result.xlsx"
            assert save_reviewed_excel(
                SAMPLE_RECEIPT,
                normal,
                human_approved=HUMAN_APPROVED,
                output_path=output_path,
                source_text=SAMPLE_OCR_TEXT,
            )

            saved = load_workbook(output_path)
            assert saved.sheetnames == ["검토_요약", "품목", "원문"]
            assert [cell.value for cell in saved["검토_요약"][1]] == [
                "field",
                "raw_value",
                "cleaned_value",
                "final_value",
                "review_status",
            ]
            print("저장 완료:", output_path, saved.sheetnames)
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

            이전 앱 없이 내장 `SAMPLE_RECEIPT`를 같은 검증·Excel 함수에 전달합니다.
            """
        ),
    ]
    return notebook("07_validation_export.ipynb", cells)


def notebook_08() -> dict:
    cells = [
        intro(
            8,
            "실무 적용 시나리오 설계 및 최종 정리",
            "poc_candidate_card.md",
            "견적서·신청서·거래명세서 중 하나의 PoC 조건을 정리합니다.",
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
            | 저장 형식과 위치 | 승인 후 Excel, 승인된 저장소 |
            | 원본·결과 삭제 시점 | 조직의 보존 기준에 따름 |
            | PoC 중단 조건 | 수정률이 허용 범위를 넘으면 중단 |

            ## 적용 전 확인

            - [x] 합성 문서로 기능을 점검했다.
            - [ ] 실제 개인정보의 외부 전송 승인을 확인한다.
            - [ ] 최종 승인자와 반려 절차를 확인한다.
            '''

            output_path = OUTPUT_DIR / "poc_candidate_card.md"
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
    5: "streamlit_basic",
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
