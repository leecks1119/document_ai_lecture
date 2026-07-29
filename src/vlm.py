"""4교시: PaddleOCR-VL 1.6 문서 파싱 선택 경로."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .sample_data import SAMPLE_VLM_RESULT


def load_mock_vlm() -> dict:
    """모델 다운로드 없이 사용할 합성 VLM 문서 파싱 결과를 반환한다."""

    return deepcopy(SAMPLE_VLM_RESULT)


def vlm_text_from_result(result: dict) -> str:
    """페이지별 Markdown을 정보 추출용 텍스트로 합친다."""

    return "\n\n".join(
        page.get("markdown", "")
        for page in result.get("pages", [])
        if page.get("markdown")
    )


def parse_with_paddleocr_vl(
    document_path: str | Path,
    *,
    engine: str = "transformers",
) -> dict:
    """PaddleOCR-VL 1.6 전체 파이프라인으로 문서를 파싱한다."""

    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR-VL 의존성이 없습니다. "
            "requirements-vlm.txt를 설치한 뒤 다시 실행하세요."
        ) from exc

    path = Path(document_path)
    if not path.is_file():
        raise ValueError(f"문서 파일을 찾을 수 없습니다: {path}")

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        engine=engine,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    pages = []
    for page_number, page_result in enumerate(
        pipeline.predict(str(path)),
        start=1,
    ):
        payload = getattr(page_result, "json", {})
        if callable(payload):
            payload = payload()
        page_data = payload.get("res", payload)

        markdown_payload = getattr(page_result, "markdown", {}) or {}
        if callable(markdown_payload):
            markdown_payload = markdown_payload()
        markdown_text = (
            markdown_payload.get("markdown_texts")
            or markdown_payload.get("text")
            or page_data.get("markdown", "")
        )
        if isinstance(markdown_text, list):
            markdown_text = "\n\n".join(map(str, markdown_text))

        blocks = []
        for block in page_data.get("parsing_res_list", []):
            blocks.append(
                {
                    "label": block.get("block_label"),
                    "content": block.get("block_content"),
                    "order": block.get("block_order"),
                }
            )
        pages.append(
            {
                "page": page_number,
                "markdown": str(markdown_text or ""),
                "blocks": blocks,
            }
        )

    return {
        "model_executed": True,
        "pipeline": "PaddleOCR-VL-1.6",
        "vlm_model": "PaddleOCR-VL-1.6-0.9B",
        "engine": engine,
        "source_mode": "direct_vlm",
        "input_file": path.name,
        "pages": pages,
    }
