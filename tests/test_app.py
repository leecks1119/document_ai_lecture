from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_app_builds_without_browser():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "영수증 Document AI 미니 앱"
    assert len(app.file_uploader) == 1
    assert "최대 5MB" in app.file_uploader[0].label
    assert app.file_uploader[0].proto.max_upload_size_mb == 5


def test_streamlit_sample_path_shows_result():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    app.button(key="run_sample").click().run(timeout=20)

    assert not app.exception
    assert app.info
    assert any(
        "현재 업로드한 파일을 분석한 결과가 아닙니다" in item.value
        for item in app.info
    )
    assert len(app.checkbox) == 1
    assert len(app.get("download_button")) == 0


def test_streamlit_requires_human_approval_before_download():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    app.button(key="run_sample").click().run(timeout=20)
    app.checkbox(key="review_complete").check().run(timeout=20)

    assert not app.exception
    assert len(app.get("download_button")) == 1


def test_streamlit_revalidates_learner_edits_before_download():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    app.button(key="run_sample").click().run(timeout=20)
    app.text_input(key="review_total_amount").input("999").run(timeout=20)
    app.checkbox(key="review_complete").check().run(timeout=20)

    assert not app.exception
    assert len(app.get("download_button")) == 0
    assert any("합계" in item.value for item in app.error)
