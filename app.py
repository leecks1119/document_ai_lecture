"""8교시에서 완성하는 초보자용 Gradio Document AI 미니 앱."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr

from src.export import receipt_to_rows
from src.pipeline import process_document


COURSE_NOTICE = """
### 교육용 미니 앱

- 합성 문서만 사용하세요. Colab에서 만든 Gradio 공유 주소는 공개될 수 있습니다.
- **업로드 문서 처리**는 EasyOCR가 설치된 선택 경로입니다.
- **샘플로 계속**은 업로드 문서를 읽지 않는 명시적인 mock 경로입니다.
"""


def _csv_file(csv_bytes: bytes | None) -> str | None:
    if not csv_bytes:
        return None
    with NamedTemporaryFile(
        prefix="receipt_",
        suffix=".csv",
        delete=False,
    ) as output:
        output.write(csv_bytes)
        return output.name


def _present(result: dict) -> tuple[str, str, dict, list[dict], str | None]:
    """처리 결과를 Gradio 컴포넌트 다섯 개에 맞춘다."""

    if not result.get("ok"):
        messages = "\n".join(f"- {item}" for item in result.get("errors", []))
        status = (
            f"### {result.get('status', '처리 오류')}\n{messages}\n\n"
            "원인을 확인하거나 **샘플로 계속**을 선택하세요."
        )
        return status, "", {}, [], None

    validation = result["validation"]
    status_lines = [f"### {result['status']}"]
    if validation["valid"]:
        status_lines.append("검증 결과: 저장 가능한 샘플입니다.")
    for warning in validation["warnings"]:
        status_lines.append(f"- 경고: {warning}")
    for error in validation["errors"]:
        status_lines.append(f"- 오류: {error}")

    return (
        "\n".join(status_lines),
        result["ocr_text"],
        result["data"],
        receipt_to_rows(result["data"]),
        _csv_file(result.get("csv_bytes")),
    )


def run_uploaded(file_path: str | None):
    """선택 실습: 업로드 문서를 실제 EasyOCR로 처리한다."""

    return _present(process_document(file_path))


def run_sample():
    """필수 실습: 합성 샘플의 mock 경로를 명시적으로 실행한다."""

    return _present(process_document(use_sample=True))


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="초보자용 Document AI 미니 앱") as demo:
        gr.Markdown("# 영수증 Document AI 미니 앱")
        gr.Markdown(COURSE_NOTICE)

        file_input = gr.File(
            label="합성 영수증 PNG·JPEG·PDF",
            file_types=[".png", ".jpg", ".jpeg", ".pdf"],
            type="filepath",
        )
        with gr.Row():
            live_button = gr.Button("업로드 문서 처리 · 선택")
            sample_button = gr.Button("샘플로 계속 · 기본", variant="primary")

        status = gr.Markdown()
        ocr_text = gr.Textbox(label="OCR 텍스트", lines=7)
        json_output = gr.JSON(label="구조화 JSON")
        table_output = gr.Dataframe(label="품목 표", interactive=False)
        download = gr.DownloadButton(label="검증된 CSV 다운로드")

        outputs = [status, ocr_text, json_output, table_output, download]
        live_button.click(run_uploaded, inputs=file_input, outputs=outputs)
        sample_button.click(run_sample, outputs=outputs)

    return demo


demo = build_demo()


if __name__ == "__main__":
    # 로컬 실행에서는 공개 공유 주소를 만들지 않는다.
    demo.launch(share=False, max_file_size="5mb")
