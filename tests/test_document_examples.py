from src.document_examples import (
    DOCUMENT_EXAMPLES,
    document_example,
    validate_document_example,
)


def test_all_extension_examples_are_valid():
    for data in DOCUMENT_EXAMPLES.values():
        assert validate_document_example(data)["valid"]


def test_example_returns_copy():
    data = document_example("quotation")
    data["total_amount"] = 1

    assert DOCUMENT_EXAMPLES["quotation"]["total_amount"] == 550000


def test_missing_application_field_is_blocked():
    data = document_example("application")
    data["manager_approval"] = None

    result = validate_document_example(data)

    assert not result["valid"]
    assert "필수값 누락: manager_approval" in result["errors"]
