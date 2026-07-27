"""Streamlit Document AI 미니 앱.

학습자는 Colab에서 이 파일을 만들고 Streamlit AppTest로 기능을 확인한다.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st

from src.export import receipt_to_rows, receipt_to_xlsx_bytes
from src.pipeline import process_document


st.set_page_config(page_title="영수증 Document AI", layout="wide")
st.title("영수증 Document AI 미니 앱")
st.caption("한 번에 문서 한 장 · 원문 대조 후 Excel 저장")
st.warning(
    "카드번호·승인번호·전화번호 등 식별정보를 먼저 가리세요. "
    "개인 문서는 외부 AI API에 전송하지 않습니다."
)

uploaded_file = st.file_uploader(
    "영수증 이미지 또는 PDF 한 장",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=False,
)
processor = st.radio(
    "처리 경로",
    options=["ocr", "vlm"],
    format_func=lambda value: (
        "PaddleOCR · PP-OCRv5 Korean"
        if value == "ocr"
        else "문서 VLM · 강사 시연/준비 결과"
    ),
)

left, right = st.columns(2)
run_uploaded = left.button("업로드 문서 처리", type="primary", key="run_uploaded")
run_sample = right.button("준비 결과로 실행", key="run_sample")


def _run_uploaded_file() -> dict:
    if uploaded_file is None:
        return process_document(None, processor=processor)

    suffix = Path(uploaded_file.name).suffix.lower()
    with NamedTemporaryFile(suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file.flush()
        return process_document(temp_file.name, processor=processor)


if run_uploaded:
    st.session_state["result"] = _run_uploaded_file()
    st.session_state["review_complete"] = False
elif run_sample:
    st.session_state["result"] = process_document(
        processor=processor,
        use_sample=True,
    )
    st.session_state["review_complete"] = False

result = st.session_state.get("result")
if result:
    if not result.get("ok"):
        st.error(result.get("status", "처리 오류"))
        for message in result.get("errors", []):
            st.write(f"- {message}")
        st.info("원인을 확인하거나 준비 결과로 실행하세요.")
    else:
        st.success(result["status"])
        validation = result["validation"]
        for warning in validation["warnings"]:
            st.warning(warning)
        for error in validation["errors"]:
            st.error(error)

        tab_text, tab_json, tab_table = st.tabs(
            ["판독 원문", "구조화 JSON", "품목 표"]
        )
        with tab_text:
            st.text_area("OCR·문서 판독 결과", result["ocr_text"], height=240)
        with tab_json:
            st.json(result["data"])
        with tab_table:
            st.dataframe(
                pd.DataFrame(receipt_to_rows(result["data"])),
                width="stretch",
                hide_index=True,
            )

        review_complete = st.checkbox(
            "원본 영수증과 추출값을 직접 대조했고 이 결과를 승인합니다.",
            key="review_complete",
        )
        if validation["valid"] and review_complete:
            xlsx_bytes = receipt_to_xlsx_bytes(
                result["data"],
                source_text=result["ocr_text"],
                review_status="사람 확인 완료",
            )
            st.download_button(
                "검증된 Excel 다운로드",
                data=xlsx_bytes,
                file_name="receipt_result.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )
        elif validation["errors"]:
            st.warning("검증 오류가 남아 있어 Excel 다운로드를 차단했습니다.")
        else:
            st.info("원본을 확인하고 승인해야 Excel을 다운로드할 수 있습니다.")
