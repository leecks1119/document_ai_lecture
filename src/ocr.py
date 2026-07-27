"""2교시: 준비된 OCR 결과와 선택 EasyOCR 경로."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from .sample_data import SAMPLE_OCR_RESULT


def load_mock_ocr() -> list[dict]:
    """API 키나 모델 다운로드 없이 사용할 수업용 OCR 결과를 반환한다."""

    return deepcopy(SAMPLE_OCR_RESULT)


def ocr_text_from_result(result: list[dict]) -> str:
    """OCR 결과의 텍스트를 읽기 순서대로 합친다."""

    return "\n".join(item["text"] for item in result)


def extract_with_easyocr(
    image_path: str | Path,
    languages: tuple[str, ...] = ("ko", "en"),
) -> list[dict]:
    """선택 실습: EasyOCR 결과를 공통 형식으로 바꾼다.

    EasyOCR가 설치되지 않았거나 모델 다운로드가 막히면 예외를 그대로
    전달한다. 호출하는 쪽에서 오류를 먼저 보여 준 뒤 사용자가 명시적으로
    mock 경로를 선택해야 한다.
    """

    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR가 설치되지 않았습니다. '샘플로 계속'을 선택하세요."
        ) from exc

    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"이미지 파일을 찾을 수 없습니다: {path}")

    reader = easyocr.Reader(list(languages), gpu=False)

    with TemporaryDirectory(prefix="docai_ocr_") as temp_dir:
        image_paths = _prepare_image_paths(path, Path(temp_dir))
        result: list[dict] = []
        for page_number, image_path in enumerate(image_paths, start=1):
            raw_page = reader.readtext(str(image_path))
            result.extend(
                {
                    "page": page_number,
                    "box": [[int(x), int(y)] for x, y in box],
                    "text": text,
                    "confidence": float(confidence),
                }
                for box, text, confidence in raw_page
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
