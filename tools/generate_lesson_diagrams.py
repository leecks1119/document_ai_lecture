"""교시별 핵심 개념을 설명하는 재현 가능한 SVG 도식을 만든다."""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "lessons" / "assets"

COLORS = {
    "navy": "#173B57",
    "teal": "#167D7F",
    "mint": "#DDF3EE",
    "cream": "#FFF9EF",
    "amber": "#F4B740",
    "red": "#C23B32",
    "green": "#248A5B",
    "gray": "#667085",
    "line": "#CBD5E1",
    "white": "#FFFFFF",
}


def text_lines(
    x: int,
    y: int,
    lines: list[str],
    *,
    size: int = 28,
    weight: int = 600,
    color: str = "#173B57",
    anchor: str = "middle",
    gap: int = 38,
) -> str:
    tspans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else gap}">{escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{tspans}</text>'
    )


def svg_frame(title: str, description: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="675" viewBox="0 0 1200 675" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(description)}</desc>
  <rect width="1200" height="675" fill="{COLORS['cream']}"/>
  <rect x="36" y="32" width="1128" height="611" rx="28" fill="{COLORS['white']}" stroke="{COLORS['line']}" stroke-width="2"/>
  <text x="76" y="92" font-family="Arial, Apple SD Gothic Neo, Noto Sans KR, sans-serif" font-size="34" font-weight="700" fill="{COLORS['navy']}">{escape(title)}</text>
  <g font-family="Arial, Apple SD Gothic Neo, Noto Sans KR, sans-serif">
    {body}
  </g>
  <text x="76" y="615" font-family="Arial, Apple SD Gothic Neo, Noto Sans KR, sans-serif" font-size="20" fill="{COLORS['gray']}">Document AI 입문 과정 · 2026</text>
</svg>
"""


def arrow(x1: int, y: int, x2: int, color: str = "#167D7F") -> str:
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 16}" y2="{y}" '
        f'stroke="{color}" stroke-width="6" stroke-linecap="round"/>'
        f'<path d="M {x2 - 20} {y - 12} L {x2} {y} L {x2 - 20} {y + 12}" '
        f'fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
    )


def flow_diagram(
    title: str,
    description: str,
    nodes: list[tuple[str, str]],
    footer: str,
) -> str:
    count = len(nodes)
    gap = 32
    width = min(190, (1010 - gap * (count - 1)) // count)
    total = width * count + gap * (count - 1)
    start_x = (1200 - total) // 2
    y = 215
    body = []
    for index, (label, note) in enumerate(nodes):
        x = start_x + index * (width + gap)
        body.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="190" rx="22" '
            f'fill="{COLORS["mint"] if index % 2 == 0 else "#EAF1F8"}" '
            f'stroke="{COLORS["teal"]}" stroke-width="2"/>'
        )
        body.append(
            f'<circle cx="{x + width // 2}" cy="{y + 48}" r="25" '
            f'fill="{COLORS["teal"]}"/>'
        )
        body.append(
            text_lines(
                x + width // 2,
                y + 57,
                [str(index + 1)],
                size=25,
                color=COLORS["white"],
            )
        )
        body.append(
            text_lines(
                x + width // 2,
                y + 108,
                label.split("|"),
                size=25,
            )
        )
        body.append(
            text_lines(
                x + width // 2,
                y + 166,
                note.split("|"),
                size=18,
                weight=400,
                color=COLORS["gray"],
                gap=26,
            )
        )
        if index < count - 1:
            body.append(arrow(x + width + 5, y + 95, x + width + gap - 5))

    body.append(
        f'<rect x="150" y="460" width="900" height="82" rx="20" '
        f'fill="{COLORS["navy"]}"/>'
    )
    body.append(
        text_lines(
            600,
            510,
            footer.split("|"),
            size=25,
            color=COLORS["white"],
            gap=31,
        )
    )
    return svg_frame(title, description, "".join(body))


def comparison_diagram(
    title: str,
    description: str,
    left_title: str,
    left_lines: list[str],
    right_title: str,
    right_lines: list[str],
    footer: str,
    *,
    left_color: str = "#EAF1F8",
    right_color: str = "#DDF3EE",
    show_arrow: bool = True,
) -> str:
    body = [
        f'<rect x="95" y="165" width="450" height="330" rx="26" fill="{left_color}" stroke="{COLORS["line"]}" stroke-width="2"/>',
        f'<rect x="655" y="165" width="450" height="330" rx="26" fill="{right_color}" stroke="{COLORS["line"]}" stroke-width="2"/>',
        text_lines(320, 220, [left_title], size=30),
        text_lines(880, 220, [right_title], size=30),
        text_lines(
            140,
            285,
            left_lines,
            size=24,
            weight=400,
            color=COLORS["gray"],
            anchor="start",
            gap=48,
        ),
        text_lines(
            700,
            285,
            right_lines,
            size=24,
            weight=400,
            color=COLORS["gray"],
            anchor="start",
            gap=48,
        ),
        f'<rect x="180" y="525" width="840" height="58" rx="18" fill="{COLORS["navy"]}"/>',
        text_lines(600, 563, [footer], size=23, color=COLORS["white"]),
    ]
    if show_arrow:
        body.insert(-2, arrow(565, 330, 635))
    return svg_frame(title, description, "".join(body))


def three_way_comparison_diagram(
    title: str,
    description: str,
    columns: list[tuple[str, list[str]]],
    footer: str,
) -> str:
    body = []
    colors = ["#EAF1F8", "#DDF3EE", "#FFF0D8"]
    for index, (heading, lines) in enumerate(columns):
        x = 72 + index * 372
        body.extend(
            [
                f'<rect x="{x}" y="155" width="330" height="340" rx="24" '
                f'fill="{colors[index]}" stroke="{COLORS["line"]}" stroke-width="2"/>',
                text_lines(x + 165, 210, [heading], size=29),
                text_lines(
                    x + 28,
                    270,
                    lines,
                    size=21,
                    weight=400,
                    color=COLORS["gray"],
                    anchor="start",
                    gap=43,
                ),
            ]
        )
    body.extend(
        [
            f'<rect x="150" y="525" width="900" height="58" rx="18" fill="{COLORS["navy"]}"/>',
            text_lines(600, 563, [footer], size=23, color=COLORS["white"]),
        ]
    )
    return svg_frame(title, description, "".join(body))


def validation_gate_diagram() -> str:
    """valid·warning·error가 병렬 판정이고 승인은 별도임을 보여 준다."""

    body = [
        f'<rect x="65" y="260" width="180" height="100" rx="22" fill="#EAF1F8" stroke="{COLORS["line"]}" stroke-width="2"/>',
        text_lines(155, 300, ["규칙 검사"], size=27),
        text_lines(155, 335, ["JSON 입력"], size=18, weight=400, color=COLORS["gray"]),
        f'<line x1="245" y1="310" x2="310" y2="310" stroke="{COLORS["navy"]}" stroke-width="5"/>',
    ]
    branches = [
        (165, COLORS["green"], "valid", "차단 오류 없음"),
        (270, COLORS["amber"], "warnings", "확인 필요"),
        (375, COLORS["red"], "errors", "수정 필요"),
    ]
    for y, color, label, note in branches:
        body.extend(
            [
                f'<polyline points="310,310 310,{y + 40} 365,{y + 40}" fill="none" stroke="{color}" stroke-width="5"/>',
                f'<rect x="365" y="{y}" width="210" height="80" rx="20" fill="{COLORS["white"]}" stroke="{color}" stroke-width="4"/>',
                text_lines(470, y + 32, [label], size=25, color=color),
                text_lines(470, y + 61, [note], size=17, weight=400, color=COLORS["gray"]),
            ]
        )
    body.extend(
        [
            f'<line x1="575" y1="205" x2="675" y2="250" stroke="{COLORS["teal"]}" stroke-width="5"/>',
            f'<line x1="575" y1="310" x2="675" y2="290" stroke="{COLORS["teal"]}" stroke-width="5"/>',
            f'<rect x="675" y="220" width="220" height="100" rx="22" fill="{COLORS["mint"]}" stroke="{COLORS["teal"]}" stroke-width="3"/>',
            text_lines(785, 262, ["사람 원본 확인"], size=24),
            text_lines(785, 294, ["승인 또는 수정"], size=17, weight=400, color=COLORS["gray"]),
            arrow(900, 270, 940),
            f'<rect x="940" y="230" width="205" height="80" rx="22" fill="{COLORS["navy"]}"/>',
            text_lines(1042, 280, ["승인 후 Excel"], size=24, color=COLORS["white"]),
            f'<line x1="575" y1="415" x2="675" y2="415" stroke="{COLORS["red"]}" stroke-width="5"/>',
            f'<rect x="675" y="377" width="220" height="76" rx="22" fill="#FFF0E8" stroke="{COLORS["red"]}" stroke-width="3"/>',
            text_lines(785, 424, ["수정 · 재검사"], size=23, color=COLORS["red"]),
            text_lines(
                600,
                555,
                ["valid=true도 사람 승인을 대신하지 않습니다."],
                size=23,
                color=COLORS["red"],
            ),
        ]
    )
    return svg_frame(
        "검증 세 갈래와 별도 승인 게이트",
        "검증 결과가 valid, warnings, errors로 갈라지고 valid 경로도 사람 원본 확인 뒤에만 Excel로 가는 그림",
        "".join(body),
    )


DIAGRAMS = {
    "02/01_ocr_anatomy.svg": comparison_diagram(
        "OCR 결과 하나를 해부하기",
        "영수증 단어 하나에 위치 좌표, 인식 텍스트, 신뢰도가 연결된 그림",
        "문서에서 본 영역",
        ["□ 바운딩 박스", "어디에서 읽었는가", "좌표 네 점"],
        "OCR가 반환한 값",
        ["텍스트: 합계 5,000원", "신뢰도: 0.99", "정답 확정은 아님"],
        "신뢰도는 정답표가 아니라 확인 순서를 정하는 신호입니다.",
    ),
    "02/02_quality_compare.svg": comparison_diagram(
        "같은 문서, 다른 입력 품질",
        "정상 이미지와 기울고 흐린 이미지에서 OCR 확인 난이도가 달라지는 비교",
        "깨끗한 입력",
        ["정면 촬영", "선명한 글자", "영역이 잘리지 않음"],
        "저품질 입력",
        ["4° 기울어짐", "글자 흐림", "일부 영역 확인 필요"],
        "입력 품질이 나쁘면 중요한 숫자를 원문에서 다시 확인합니다.",
        left_color="#DDF3EE",
        right_color="#FFF0E8",
    ),
    "03/01_clean_before_after.svg": comparison_diagram(
        "정제 전과 후",
        "OCR 원문과 공백·줄바꿈만 정리한 결과를 비교",
        "OCR 원문",
        ["합계:   5,000원", "연필  2개 × 1,000원", "불규칙한 공백"],
        "정제 결과",
        ["합계: 5,000원", "연필 2개 × 1,000원", "원문은 따로 보존"],
        "정제는 표현을 정돈할 뿐, 없는 값을 새로 만들지 않습니다.",
    ),
    "03/02_structure_map.svg": flow_diagram(
        "영수증의 네 가지 영역",
        "영수증을 헤더, 날짜, 반복 품목, 합계 영역으로 나눈 그림",
        [
            ("헤더", "상호명"),
            ("날짜", "키-값"),
            ("품목", "반복 행"),
            ("합계", "중요 필드"),
        ],
        "같은 글자라도 문서 안의 역할에 따라 다른 데이터 구조가 됩니다.",
    ),
    "04/01_receipt_to_json.svg": comparison_diagram(
        "문서 VLM 결과에서 업무 JSON으로",
        "PaddleOCR-VL의 제목과 표 Markdown을 업무 JSON 필드로 변환하는 그림",
        "VLM 중간 결과",
        ["# 상호명", "거래일자 문단", "Markdown 품목 표", "합계 강조"],
        "업무 JSON",
        ['"store_name"', '"date"', '"items": [...]', '"total_amount"'],
        "수업용 예제는 출처 기록을 남기고, 값은 원문 근거와 함께 검증합니다.",
    ),
    "04/02_three_checks.svg": flow_diagram(
        "VLM 결과의 세 가지 확인",
        "VLM 중간 결과가 구조, 자료형, 원문 근거 검사를 차례로 통과하는 그림",
        [
            ("필드 구조", "네 필드"),
            ("자료형", "날짜·정수"),
            ("원문 근거", "문서와 비교"),
        ],
        "구조가 맞는 JSON이라도 값이 정확한지는 별도로 확인합니다.",
    ),
    "05/01_component_flow.svg": flow_diagram(
        "Streamlit 화면과 함수 연결",
        "파일과 버튼 입력이 Python 함수를 거쳐 텍스트와 JSON으로 출력되는 그림",
        [
            ("파일", "입력"),
            ("OCR·VLM", "처리기 선택"),
            ("Python 함수", "처리"),
            ("텍스트·JSON", "출력"),
        ],
        "Streamlit은 이미 만든 Python 함수에 화면을 붙입니다.",
    ),
    "05/02_minimal_ui.svg": comparison_diagram(
        "최소 Streamlit 화면",
        "파일 입력, 처리기 선택, 중간 결과, JSON으로 구성된 화면",
        "입력 영역",
        ["파일 선택", "OCR·VLM 선택", "샘플로 계속"],
        "결과 영역",
        ["처리 상태", "문서 중간 결과", "JSON·품목 표"],
        "화면보다 먼저 각 Python 함수의 반환값을 확인합니다.",
    ),
    "06/01_processing_paths.svg": flow_diagram(
        "직접 읽기 실패와 수업용 예제 경로",
        "업로드한 문서를 직접 읽지 못했을 때 사용자가 수업용 예제를 선택하는 흐름",
        [
            ("업로드", "파일 확인"),
            ("직접 읽기", "PaddleOCR"),
            ("읽기 실패", "이유 표시"),
            ("예제 선택", "사용자 결정"),
            ("수업용 예제", "크게 표시"),
        ],
        "준비 결과를 실제 업로드 파일의 결과처럼 보여 주지 않습니다.",
    ),
    "06/02_status_steps.svg": comparison_diagram(
        "OCR과 VLM 선택 기준",
        "단순 문서와 복잡한 문서에서 처리 경로가 달라지는 비교",
        "PaddleOCR",
        ["단순한 글자·줄", "텍스트·위치·신뢰도", "가벼운 기본 경로"],
        "PaddleOCR-VL",
        ["표·제목·복잡한 배치", "Markdown·레이아웃", "선택 멀티모달 경로"],
        "오류 영향이 큰 값은 어느 경로든 사람 검토로 보냅니다.",
        show_arrow=False,
    ),
    "07/01_validation_signal.svg": validation_gate_diagram(),
    "07/02_json_to_excel.svg": comparison_diagram(
        "JSON 품목이 Excel 행으로",
        "JSON의 연필과 노트 품목 배열이 Excel의 두 행으로 변환되는 그림",
        "JSON items 배열",
        ["연필 · 2개", "노트 · 1개", "영수증 하나"],
        "Excel 품목 시트",
        ["1행 · 연필", "2행 · 노트", "공통 상호명·날짜"],
        "반복 품목 하나가 Excel의 한 행이 됩니다.",
    ),
    "08/01_human_review.svg": flow_diagram(
        "사람 검토가 포함된 최종 흐름",
        "문서 추출과 검증 뒤 담당자가 확인해야 Excel이 확정되는 흐름",
        [
            ("문서", "승인된 비식별"),
            ("추출", "OCR·VLM"),
            ("검증", "규칙 확인"),
            ("사람 확인", "승인·수정"),
            ("Excel 확정", "업무 활용"),
        ],
        "앱 실행이 끝이 아니라 최종 책임자와 수정 절차를 정해야 합니다.",
    ),
    "08/02_business_card.svg": comparison_diagram(
        "한 장짜리 업무 적용 카드",
        "문서 자동화 적용에 필요한 입력, 필드, 위험, 검토자, 저장, 삭제 항목",
        "무엇을 처리할까",
        ["입력 문서", "필요 필드", "오류 영향"],
        "누가 어떻게 관리할까",
        ["사람 검토자", "저장 형식·위치", "삭제 시점"],
        "새 기능보다 먼저 책임과 데이터 처리 기준을 정합니다.",
    ),
}


def main() -> None:
    for relative_path, content in DIAGRAMS.items():
        path = ASSETS / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"{len(DIAGRAMS)}개 SVG 도식을 생성했습니다.")


if __name__ == "__main__":
    main()
