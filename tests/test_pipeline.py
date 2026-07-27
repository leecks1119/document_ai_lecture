from src.pipeline import MAX_FILE_SIZE, process_document, run_smoke_test, validate_upload


def test_missing_upload_does_not_silently_fallback():
    result = process_document()

    assert not result["ok"]
    assert result["can_continue_with_sample"]
    assert "data" not in result


def test_explicit_sample_path_is_labeled():
    result = process_document(use_sample=True)

    assert result["ok"]
    assert "MOCK OCR + MOCK 추출" in result["status"]
    assert result["validation"]["valid"]


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
