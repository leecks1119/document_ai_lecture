import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request
from io import BytesIO
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "ocr-test" / "ner" / "3교시 샘플 영수증.jpg"
DEFAULT_RULE_BASED_JSON = ROOT / "ocr-test" / "ner" / "3교시_샘플_영수증_structured.json"
DEFAULT_OUTPUT_DIR = ROOT / "vlm-poc" / "outputs"
DEFAULT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "zai-org/GLM-4.5V"

# 수강생 설정:
# Hugging Face 토큰을 아래 따옴표 안에 붙여넣으면 됩니다.
# 예: HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
# 빈 문자열이면 기존처럼 환경변수 HF_TOKEN을 사용합니다.
HF_TOKEN = ""


RECEIPT_TO_DB_PROMPT = """
이 이미지는 한국어 영수증입니다. 영수증에서 보이는 데이터를 JSON으로 추출하세요.
반드시 JSON 객체 1개만 출력하세요. 마크다운, 설명, 주석은 출력하지 마세요.
보이지 않거나 확실하지 않은 값은 null로 두세요.
금액은 숫자로 변환하세요. 예: "1,600원" -> 1600
날짜와 시간은 가능하면 "YYYY-MM-DD HH:MM" 형식으로 정규화하세요.

중요:
- 상호명, 대표자명, 주소, 상품명, 영수증 유형, 결제수단은 반드시 이미지에 보이는 한글 그대로 작성하세요.
- 한글을 영어로 번역하거나 로마자로 표기하지 마세요.
- 한글 글자가 확실하지 않으면 추측해서 영어로 바꾸지 말고 null로 두세요.
- 애매한 부분은 confidence_notes에 한국어로 짧게 적으세요.

Use exactly this JSON shape:
{
  "store_name": null,
  "business_number": null,
  "representative": null,
  "phone": null,
  "address": null,
  "receipt_type": null,
  "transaction_datetime": null,
  "pos_number": null,
  "bill_number": null,
  "subtotal": null,
  "tax": null,
  "total_amount": null,
  "payment_method": null,
  "card_amount": null,
  "items": [
    {
      "name": null,
      "barcode": null,
      "unit_price": null,
      "quantity": null,
      "discount": null,
      "amount": null
    }
  ],
  "confidence_notes": []
}
""".strip()


PROMPT_EXPLANATION = """
# VLM Receipt Result
""".strip()


def ensure_utf8_stdout() -> None:
    if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass


def image_to_data_url(image_path: Path) -> str:
    try:
        from PIL import Image, ImageEnhance, ImageFilter

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            scale = max(1, min(4, 1600 // max(image.width, 1)))
            if scale > 1:
                image = image.resize(
                    (image.width * scale, image.height * scale),
                    Image.Resampling.LANCZOS,
                )
            image = ImageEnhance.Contrast(image).enhance(1.25)
            image = image.filter(ImageFilter.SHARPEN)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except ImportError:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if not mime_type:
            mime_type = "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


def call_hf_chat(endpoint: str, token: str, model: str, image_path: Path, max_tokens: int) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RECEIPT_TO_DB_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hugging Face API error {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hugging Face API call failed: {exc}") from exc

    data = json.loads(response_body)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Hugging Face response: {response_body}") from exc


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def safe_parse_json_response(text: str, raw_output_path: Path) -> dict:
    try:
        return parse_json_response(text)
    except json.JSONDecodeError as exc:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(text, encoding="utf-8")
        raise RuntimeError(
            "The VLM response was not valid JSON. "
            f"Raw response saved to: {raw_output_path.resolve()}"
        ) from exc


def load_rule_based_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp949"))
    except json.JSONDecodeError:
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_markdown_report(
    image_path: Path,
    model: str,
    vlm_result: dict,
    rule_based_result: dict | None,
    vlm_json_path: Path,
) -> str:
    fields = [
        "store_name",
        "business_number",
        "transaction_datetime",
        "total_amount",
        "payment_method",
        "bill_number",
    ]
    lines = [
        PROMPT_EXPLANATION,
        "",
        "## Input",
        "",
        f"- Image: `{image_path}`",
        f"- Model: `{model}`",
        f"- VLM JSON: `{vlm_json_path}`",
        "",
        "## Prompt",
        "",
        "```text",
        RECEIPT_TO_DB_PROMPT,
        "```",
        "",
        "## VLM Result Snapshot",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for field in fields:
        lines.append(f"| {field} | {vlm_result.get(field)} |")

    items = vlm_result.get("items") or []
    lines.extend(
        [
            "",
            "## Items",
            "",
            "| Name | Qty | Unit Price | Amount | Barcode |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in items:
        lines.append(
            "| {name} | {quantity} | {unit_price} | {amount} | {barcode} |".format(
                name=item.get("name"),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                amount=item.get("amount"),
                barcode=item.get("barcode"),
            )
        )

    notes = vlm_result.get("confidence_notes") or []
    lines.extend(["", "## Confidence Notes", ""])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None")

    lines.extend(["", "## Existing Rule-Based Result", ""])
    if rule_based_result is None:
        lines.append("Existing structured JSON was not found or could not be parsed.")
    else:
        rule_keys = ", ".join(rule_based_result.keys())
        lines.append(f"Top-level keys from existing result: `{rule_keys}`")
    return "\n".join(lines)


def main() -> int:
    ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Lesson demo: replace OCR box rules with a Hugging Face VLM prompt."
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Receipt image path")
    parser.add_argument(
        "--rule-json",
        default=str(DEFAULT_RULE_BASED_JSON),
        help="Existing rule-based structured JSON path",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--model", default=os.getenv("HF_MODEL", DEFAULT_MODEL))
    parser.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--print-prompt-only",
        action="store_true",
        help="Print the VLM prompt without calling the API.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    rule_json_path = Path(args.rule_json)
    out_dir = Path(args.out_dir)

    if args.print_prompt_only:
        print(RECEIPT_TO_DB_PROMPT)
        return 0

    token = HF_TOKEN.strip() or os.getenv(args.token_env)
    if not token:
        print(f"[!] Set HF_TOKEN at the top of this file or set {args.token_env}.", file=sys.stderr)
        print('    Source edit example: HF_TOKEN = "hf_your_token_here"', file=sys.stderr)
        print(f"    PowerShell example: $env:{args.token_env}='hf_your_token_here'", file=sys.stderr)
        print("    To preview the lesson prompt only, run with --print-prompt-only.", file=sys.stderr)
        return 1

    if not image_path.exists():
        print(f"[!] Image not found: {image_path}", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vlm_json_path = out_dir / f"receipt_vlm_result_{timestamp}.json"
    report_path = out_dir / f"receipt_vlm_lesson_report_{timestamp}.md"
    raw_output_path = out_dir / f"receipt_vlm_raw_response_{timestamp}.txt"

    print(f"[+] image : {image_path.resolve()}")
    print(f"[+] model : {args.model}")
    print("[+] calling Hugging Face VLM...")

    content = call_hf_chat(args.endpoint, token, args.model, image_path, args.max_tokens)
    vlm_result = safe_parse_json_response(content, raw_output_path)
    rule_based_result = load_rule_based_result(rule_json_path)

    write_json(vlm_json_path, vlm_result)
    report = make_markdown_report(
        image_path=image_path,
        model=args.model,
        vlm_result=vlm_result,
        rule_based_result=rule_based_result,
        vlm_json_path=vlm_json_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"[+] VLM JSON saved : {vlm_json_path.resolve()}")
    print(f"[+] Lesson report  : {report_path.resolve()}")
    print(f"[+] Store          : {vlm_result.get('store_name')}")
    print(f"[+] Total amount   : {vlm_result.get('total_amount')}")
    print(f"[+] Items          : {len(vlm_result.get('items') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
