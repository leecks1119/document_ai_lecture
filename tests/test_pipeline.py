from src.pipeline import process_document, run_smoke_test


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
