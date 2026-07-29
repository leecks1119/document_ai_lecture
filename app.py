"""Streamlit Document AI 미니 앱.

학습자는 Colab에서 이 파일을 만들고 Streamlit AppTest로 기능을 확인한다.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.export import receipt_to_rows, receipt_to_xlsx_bytes
from src.pipeline import process_document
from src.validate import validate_receipt


st.set_page_config(page_title="영수증 Document AI", layout="wide")
st.title("영수증 Document AI 미니 앱")
st.caption("한 번에 문서 한 장 · 원문 대조 후 Excel 저장")
st.warning(
    "Google Colab도 외부 클라우드입니다. 조직 승인 없는 개인·회사 문서는 "
    "업로드하지 마세요. 필수 실습은 공개·합성 샘플로 진행합니다."
)


def editor_integer(value):
    """표 편집값을 정수일 때만 변환하고 소수 입력은 검증기에 남긴다."""

    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and float(value).is_integer():
        return int(value)
    return value

uploaded_file = st.file_uploader(
    "영수증 이미지 또는 PDF 한 장 · 최대 5MB",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=False,
    max_upload_size=5,
    help="PNG, JPG, JPEG, PDF만 허용합니다. 수업에서는 한 번에 5MB 이하 한 장만 처리합니다.",
)
processor = st.radio(
    "처리 경로",
    options=["ocr", "vlm"],
    format_func=lambda value: (
        "PaddleOCR · PP-OCRv5 Korean"
        if value == "ocr"
        else "PaddleOCR-VL 1.6 · 실제 모델(설치·GPU 필요)"
    ),
)

left, right = st.columns(2)
run_uploaded = left.button("내 영수증 직접 읽기", type="primary", key="run_uploaded")
run_sample = right.button("수업용 예제로 계속하기", key="run_sample")


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
        st.info("원인을 확인하거나 아래 ‘수업용 예제로 계속하기’를 누르세요.")
    else:
        is_course_example = (
            result.get("data", {})
            .get("provenance", {})
            .get("engine")
            == "not_executed"
        )
        if is_course_example:
            st.info(result["status"])
        else:
            st.success(result["status"])

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

        st.subheader("원본 대조 후 수정")
        st.caption(
            "AI 결과를 그대로 승인하지 말고 원본 영수증과 비교해 수정하세요. "
            "수정값으로 다시 검증한 뒤 Excel을 만듭니다."
        )
        reviewed_data = deepcopy(result["data"])
        field_left, field_middle, field_right = st.columns(3)
        reviewed_data["store_name"] = field_left.text_input(
            "상호명",
            value=str(reviewed_data.get("store_name") or ""),
            key="review_store_name",
        ).strip() or None
        reviewed_data["date"] = field_middle.text_input(
            "날짜 · YYYY-MM-DD",
            value=str(reviewed_data.get("date") or ""),
            key="review_date",
        ).strip() or None
        total_text = field_right.text_input(
            "총액 · 숫자만",
            value=(
                str(reviewed_data["total_amount"])
                if isinstance(reviewed_data.get("total_amount"), int)
                and not isinstance(reviewed_data.get("total_amount"), bool)
                else ""
            ),
            key="review_total_amount",
        )
        try:
            reviewed_data["total_amount"] = int(
                total_text.replace(",", "").strip()
            )
        except ValueError:
            reviewed_data["total_amount"] = None

        editable_items = pd.DataFrame(
            reviewed_data.get("items") or [],
            columns=["name", "quantity", "unit_price", "line_total"],
        )
        edited_items = st.data_editor(
            editable_items,
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            key="review_items",
            column_config={
                "name": st.column_config.TextColumn("품목명"),
                "quantity": st.column_config.NumberColumn("수량", min_value=0, step=1),
                "unit_price": st.column_config.NumberColumn(
                    "단가", min_value=0, step=1
                ),
                "line_total": st.column_config.NumberColumn(
                    "금액", min_value=0, step=1
                ),
            },
        )
        reviewed_data["items"] = [
            {
                "name": str(row.get("name") or "").strip() or None,
                "quantity": editor_integer(row.get("quantity")),
                "unit_price": editor_integer(row.get("unit_price")),
                "line_total": editor_integer(row.get("line_total")),
            }
            for row in edited_items.to_dict("records")
        ]

        validation = validate_receipt(reviewed_data)
        if validation["valid"]:
            st.success("규칙 검증 통과 · 사람 승인 대기")
        for warning in validation["warnings"]:
            st.warning(warning)
        for error in validation["errors"]:
            st.error(error)

        reviewer = st.text_input(
            "검토자 ID 또는 교육용 이름",
            value="learner",
            key="reviewer",
        )
        review_note = st.text_input(
            "수정·확인 메모",
            value="원본의 상호명·날짜·품목·총액 대조 완료",
            key="review_note",
        )
        review_complete = st.checkbox(
            "원본 영수증과 추출값을 직접 대조했고 이 결과를 승인합니다.",
            key="review_complete",
        )
        if validation["valid"] and review_complete and reviewer.strip():
            decision = (
                "CHANGED"
                if reviewed_data != result["data"]
                else "APPROVED"
            )
            review_record = {
                "decision": decision,
                "reviewer": reviewer.strip(),
                "reviewed_at": datetime.now(timezone.utc).astimezone().isoformat(
                    timespec="seconds"
                ),
                "note": review_note.strip(),
            }
            xlsx_bytes = receipt_to_xlsx_bytes(
                reviewed_data,
                source_text=result["ocr_text"],
                review_status=decision,
                review_record=review_record,
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
        elif review_complete and not reviewer.strip():
            st.info("승인 기록에 남길 검토자 ID를 입력하세요.")
        else:
            st.info("원본을 확인하고 승인해야 Excel을 다운로드할 수 있습니다.")
