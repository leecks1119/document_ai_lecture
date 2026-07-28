"""공식 배포처에서 수업용 공개 실물 영수증을 재현한다."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RECEIPTS_DIR = ROOT / "sample_docs" / "public_receipts"
CORD_OUTPUT_DIR = PUBLIC_RECEIPTS_DIR / "cord_v2"
KOREA_OUTPUT_DIR = PUBLIC_RECEIPTS_DIR / "korea"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASET = "naver-clova-ix/cord-v2"
KOREAN_RECEIPT_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/7/76/"
    "Receipt_taebaek_restaurant_IMG_2614_modified.jpg"
)
KOREAN_SOURCE_SHA256 = (
    "f7251ffaeab8a56aed534f1295058117b09c4b41c85bffcc7c1a7445c6d89433"
)
KOREAN_FILENAME = "taebaek_restaurant_2025_redacted.png"
KOREAN_DERIVATIVE_SHA256 = (
    "19227c7298a16ee69bef2d7bed65826b8a1cba5389375e4ae77d02005362641f"
)

SAMPLES = [
    (
        0,
        "cord_v2_test_000.jpg",
        "4e13b815b540420a4d68229494bd6259768c226a595114609970edae9a620aa9",
    ),
    (
        1,
        "cord_v2_test_001.jpg",
        "636284664cfc357e05f7b10d67ddb380f828a1be0a20809514f9e7fc0c611a9f",
    ),
    (
        2,
        "cord_v2_test_002.jpg",
        "bcf0c175181bfc6b2352ab8f452704b8f18e27fa51f2278bc5c41f6765179edb",
    ),
]


def read_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "document-ai-course/2026"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def sample_url(row: int) -> str:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "test",
            "offset": row,
            "length": 1,
        }
    )
    payload = json.loads(read_url(f"{ROWS_ENDPOINT}?{query}"))
    return payload["rows"][0]["row"]["image"]["src"]


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_korean_receipt_derivative(content: bytes) -> bytes:
    """전화번호·거래 식별 영역을 가리고 내장 메타데이터를 제거한다."""

    if sha256(content) != KOREAN_SOURCE_SHA256:
        raise RuntimeError("한국 영수증 원본 SHA-256 불일치")

    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")

    # 연락처와 내부 거래번호가 함께 있는 행을 가린다.
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 760, 2350, 990), fill="#FFFFFF")

    # 하단 결제 식별 정보 영역은 실습에 필요하지 않아 제외한다.
    image = image.crop((0, 0, 2558, 2850))

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def download_korean_receipt() -> None:
    KOREA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = KOREA_OUTPUT_DIR / KOREAN_FILENAME
    derivative = build_korean_receipt_derivative(read_url(KOREAN_RECEIPT_URL))
    if sha256(derivative) != KOREAN_DERIVATIVE_SHA256:
        raise RuntimeError("한국 영수증 파생본 SHA-256 불일치")
    target.write_bytes(derivative)
    with Image.open(target) as image:
        image.verify()
    print("생성:", target.relative_to(ROOT), sha256(derivative))


def download_cord_receipts() -> None:
    CORD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for row, filename, expected_hash in SAMPLES:
        target = CORD_OUTPUT_DIR / filename
        if target.is_file() and sha256(target.read_bytes()) == expected_hash:
            print("확인:", target.relative_to(ROOT))
            continue

        content = read_url(sample_url(row))
        actual_hash = sha256(content)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{filename}: SHA-256 불일치 "
                f"(expected={expected_hash}, actual={actual_hash})"
            )

        target.write_bytes(content)
        with Image.open(target) as image:
            image.verify()
        print("다운로드:", target.relative_to(ROOT))

    print("CORD v2 해외 비교 영수증 3장 검증 완료")


def main() -> None:
    download_korean_receipt()
    download_cord_receipts()
    print("공개 실물 영수증 재현 완료")


if __name__ == "__main__":
    main()
