from src.pipeline import MAX_FILE_SIZE, process_document, run_smoke_test, validate_upload


def test_missing_upload_does_not_silently_fallback():
    result = process_document()

    assert not result["ok"]
    assert result["can_continue_with_sample"]
    assert "data" not in result


def test_explicit_sample_path_is_labeled():
    result = process_document(use_sample=True)

    assert result["ok"]
    assert "PREPARED REPLAY" in result["status"]
    assert result["validation"]["valid"]
    assert result["review_status"] == "PENDING_REVIEW"
    assert result["xlsx_bytes"] is None
    assert result["data"]["total_amount"] == 76000
    assert result["data"]["items"][2]["name"] == "수제 돈가스"
    assert all(item["name"] != "숙제 돈가스" for item in result["data"]["items"])
    assert result["data"]["source_mode"] == "prepared_fixture_rule_extraction"


def test_explicit_vlm_sample_is_labeled_and_structured():
    result = process_document(use_sample=True, processor="vlm")

    assert result["ok"]
    assert "VLM 구조 시연 fixture" in result["status"]
    assert result["data"]["total_amount"] == 76000
    assert len(result["data"]["items"]) == 5
    assert result["data"]["provenance"]["engine"] == "not_executed"
    assert (
        result["data"]["source_mode"]
        == "prepared_vlm_structure_fixture_rule_extraction"
    )


def test_human_approval_enables_xlsx():
    result = process_document(use_sample=True, human_approved=True)

    assert result["review_status"] == "APPROVED"
    assert result["xlsx_bytes"]


def test_explicit_review_record_is_preserved():
    review = {
        "decision": "CHANGED",
        "reviewer": "learner-01",
        "reviewed_at": "2026-07-28T15:00:00+09:00",
        "note": "원본 확인",
    }
    result = process_document(use_sample=True, review_record=review)

    assert result["review_status"] == "CHANGED"
    assert result["review_record"] == review
    assert result["xlsx_bytes"]


def test_smoke_test_covers_expected_paths():
    assert all(run_smoke_test().values())


def test_upload_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "receipt.exe"
    path.write_bytes(b"not an image")

    assert validate_upload(path) == ["PNG, JPEG, PDF 파일만 사용할 수 있습니다."]


def test_upload_rejects_files_over_course_limit(tmp_path):
    path = tmp_path / "large.png"
    path.write_bytes(b"0" * (MAX_FILE_SIZE + 1))

    assert validate_upload(path) == ["수업에서는 5MB 이하 파일만 사용합니다."]


def test_upload_rejects_multipage_pdf(tmp_path):
    fitz = __import__("fitz")
    path = tmp_path / "two_pages.pdf"
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(path)
    document.close()

    assert validate_upload(path) == ["필수 실습은 PDF 한 페이지만 처리합니다."]


def test_invalid_live_ocr_result_does_not_create_xlsx(tmp_path, monkeypatch):
    path = tmp_path / "receipt.png"
    path.write_bytes(b"synthetic image placeholder")
    monkeypatch.setattr(
        "src.pipeline.extract_with_paddleocr",
        lambda _: [
            {
                "page": 1,
                "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "text": text,
                "confidence": 0.9,
            }
            for text in (
                "샘플문구점",
                "거래일자: 2026-07-27",
                "연필 2개 × 1,000원 = 2,000원",
                "합계: 2,00o원",
            )
        ],
    )

    result = process_document(path)

    assert result["ok"]
    assert not result["validation"]["valid"]
    assert result["xlsx_bytes"] is None


def test_live_vlm_path_uses_paddleocr_vl_markdown(tmp_path, monkeypatch):
    path = tmp_path / "receipt.png"
    path.write_bytes(b"synthetic image placeholder")
    monkeypatch.setattr(
        "src.pipeline.parse_with_paddleocr_vl",
        lambda _: {
            "model": "PaddleOCR-VL-1.6",
            "pages": [
                {
                    "page": 1,
                    "markdown": (
                        "# 샘플문구점\n거래일자: 2026-07-27\n"
                        "| 연필 | 2 | 1,000원 | 2,000원 |\n"
                        "**합계: 2,000원**"
                    ),
                    "blocks": [],
                }
            ],
        },
    )

    result = process_document(path, processor="vlm")

    assert result["ok"]
    assert result["validation"]["valid"]
    assert "PaddleOCR-VL 1.6" in result["status"]
