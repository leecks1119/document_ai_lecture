"""2교시: 준비된 결과와 PaddleOCR 3.7 선택 경로."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from .sample_data import SAMPLE_OCR_RESULT


def load_mock_ocr() -> list[dict]:
    """API 키나 모델 다운로드 없이 사용할 수업용 OCR 결과를 반환한다."""

    return deepcopy(SAMPLE_OCR_RESULT)


def reconstruct_spatial_lines(result: list[dict]) -> list[str]:
    """OCR 토큰을 페이지·y좌표·x좌표 기준의 읽기 행으로 복원한다.

    PaddleOCR는 영수증의 ``품목명 / 단가 / 수량 / 금액`` 한 행을 여러
    토큰으로 반환할 수 있다. 단순 줄바꿈으로 합치면 행 관계가 사라지므로,
    각 토큰 중심 y좌표가 가까운 것끼리 묶고 x좌표 순서로 정렬한다.
    위치가 없는 준비 결과는 입력 순서를 보존한다.
    """

    positioned_by_page: dict[int, list[dict]] = defaultdict(list)
    unpositioned_by_page: dict[int, list[tuple[int, str]]] = defaultdict(list)

    for order, item in enumerate(result):
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        page = int(item.get("page") or 1)
        points = [
            point
            for point in (item.get("box") or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if not points:
            unpositioned_by_page[page].append((order, text))
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        positioned_by_page[page].append(
            {
                "text": text,
                "x": min(xs),
                "y": sum(ys) / len(ys),
                "height": max(ys) - min(ys),
                "order": order,
            }
        )

    pages = sorted(set(positioned_by_page) | set(unpositioned_by_page))
    lines: list[str] = []
    for page in pages:
        rows: list[dict] = []
        for token in sorted(
            positioned_by_page[page],
            key=lambda value: (value["y"], value["x"], value["order"]),
        ):
            row = rows[-1] if rows else None
            tolerance = (
                max(
                    12.0,
                    min(24.0, max(row["height"], token["height"]) * 0.45),
                )
                if row
                else 12.0
            )
            if row and abs(token["y"] - row["y"]) <= tolerance:
                row["tokens"].append(token)
                count = len(row["tokens"])
                row["y"] = (row["y"] * (count - 1) + token["y"]) / count
                row["height"] = max(row["height"], token["height"])
            else:
                rows.append(
                    {
                        "tokens": [token],
                        "y": token["y"],
                        "height": token["height"],
                    }
                )

        lines.extend(
            " ".join(
                token["text"]
                for token in sorted(
                    row["tokens"],
                    key=lambda value: (value["x"], value["order"]),
                )
            )
            for row in rows
        )
        lines.extend(
            text
            for _, text in sorted(
                unpositioned_by_page[page],
                key=lambda value: value[0],
            )
        )
    return lines


def ocr_text_from_result(result: list[dict]) -> str:
    """OCR 결과를 표 행 관계가 보존된 읽기 순서 텍스트로 합친다."""

    return "\n".join(reconstruct_spatial_lines(result))


def extract_with_paddleocr(
    image_path: str | Path,
    *,
    lang: str = "korean",
    ocr_version: str = "PP-OCRv5",
) -> list[dict]:
    """PaddleOCR 3.x 결과를 수업의 공통 형식으로 바꾼다.

    PP-OCRv6는 현재 한국어 모델을 제공하지 않으므로 한국어 영수증에는
    PP-OCRv5 Korean을 명시한다. 설치나 모델 다운로드가 막히면 예외를 그대로
    전달한다. 호출하는 쪽에서 오류를 먼저 보여 준 뒤 사용자가 명시적으로
    mock 경로를 선택해야 한다.
    """

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR가 설치되지 않았습니다. '샘플로 계속'을 선택하세요."
        ) from exc

    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"이미지 파일을 찾을 수 없습니다: {path}")

    pipeline = PaddleOCR(
        lang=lang,
        ocr_version=ocr_version,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )

    with TemporaryDirectory(prefix="docai_ocr_") as temp_dir:
        image_paths = _prepare_image_paths(path, Path(temp_dir))
        result: list[dict] = []
        for page_number, image_path in enumerate(image_paths, start=1):
            for page_result in pipeline.predict(str(image_path)):
                payload = getattr(page_result, "json", page_result)
                if callable(payload):
                    payload = payload()
                page_data = payload.get("res", payload)
                texts = page_data.get("rec_texts", [])
                scores = page_data.get("rec_scores", [])
                boxes = page_data.get("rec_polys", [])
                for box, text, confidence in zip(boxes, texts, scores):
                    points = box.tolist() if hasattr(box, "tolist") else box
                    result.append(
                        {
                            "page": page_number,
                            "box": [
                                [int(point[0]), int(point[1])]
                                for point in points
                            ],
                            "text": str(text),
                            "confidence": float(confidence),
                        }
                    )
        return result


def _prepare_image_paths(path: Path, temp_dir: Path) -> list[Path]:
    """이미지는 그대로 사용하고 PDF는 최대 세 페이지를 PNG로 바꾼다."""

    if path.suffix.lower() != ".pdf":
        return [path]

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF 변환 패키지가 없습니다. PNG 샘플로 계속하세요."
        ) from exc

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ValueError("PDF 파일을 열 수 없습니다.") from exc

    try:
        if document.needs_pass:
            raise ValueError("암호가 설정된 PDF는 수업에서 처리하지 않습니다.")
        if len(document) > 3:
            raise ValueError("수업에서는 PDF를 최대 3페이지만 처리합니다.")

        image_paths = []
        for page_index, page in enumerate(document):
            output_path = temp_dir / f"page_{page_index + 1}.png"
            page.get_pixmap(dpi=200, alpha=False).save(output_path)
            image_paths.append(output_path)
        return image_paths
    finally:
        document.close()
