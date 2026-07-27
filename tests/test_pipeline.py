from src.pipeline import MAX_FILE_SIZE, process_document, run_smoke_test, validate_upload


def test_missing_upload_does_not_silently_fallback():
    result = process_document()

    assert not result["ok"]
    assert result["can_continue_with_sample"]
    assert "data" not in result


def test_explicit_sample_path_is_labeled():
    result = process_document(use_sample=True)

    assert result["ok"]
    assert "MOCK PaddleOCR + MOCK VLM + MOCK 추출" in result["status"]
    assert result["validation"]["valid"]
    assert result["review_status"] == "PENDING_REVIEW"
    assert result["xlsx_bytes"] is None


def test_human_approval_enables_xlsx():
    result = process_document(use_sample=True, human_approved=True)

    assert result["review_status"] == "APPROVED"
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
