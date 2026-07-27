# 과정 참고자료와 적용 범위

조사 기준일: 2026-07-27
검토 범위: 공식 제품 문서, 표준, 규제기관 자료, 공공기관 지침

이 과정은 참고 링크를 장식처럼 나열하지 않습니다. 아래 21개 자료를 실제로 검토하고, 각 자료가 뒷받침하는 주장과 적용 교시를 구분했습니다. 18개는 채택하고 3개는 적용 범위를 제한해 조건부로 채택했습니다.

## 교시별 핵심 근거

| # | 적용 교시 | 기관·자료 | 교재에 적용한 내용 | 판정·주의 |
| ---: | --- | --- | --- | --- |
| 1 | 1·6·8 | [Google Cloud Document AI 개요](https://docs.cloud.google.com/document-ai/docs/overview) | OCR·분류·분할·필드 추출을 거쳐 비정형 문서를 구조화하는 흐름 | 채택 |
| 2 | 1 | [AWS Intelligent Document Processing 설명](https://aws.amazon.com/what-is/intelligent-document-processing/) | 문서 입력을 자동화해 디지털 업무 프로세스에 연결하는 IDP 범위 | 조건부: 정의와 단계만 사용하고 홍보성 성능 표현은 제외 |
| 3 | 2·6 | [Google Cloud Document AI 지원 파일](https://docs.cloud.google.com/document-ai/docs/file-types) | 촬영·스캔 품질이 OCR에 영향을 준다는 입력 품질 안내 | 채택: 특정 해상도 권장을 모든 OCR의 절대 기준으로 일반화하지 않음 |
| 4 | 1·2·3 | [Google Cloud Document AI 응답 처리](https://docs.cloud.google.com/document-ai/docs/handle-response) | OCR 결과에 텍스트 외에도 위치·레이아웃·정규화 값·신뢰도가 포함될 수 있음 | 채택 |
| 5 | 2·6 | [PyPI PaddleOCR](https://pypi.org/project/paddleocr/) | 수업에서 사용하는 PaddleOCR 3.7.0 버전과 Python 호환 확인 | 채택 |
| 6 | 2 | [PP-OCRv5 다국어 인식](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html) | 한국어 영수증에 `lang="korean"`과 PP-OCRv5 Korean 설정 사용 | 채택 |
| 7 | 2·3·6 | [PaddleOCR OCR 파이프라인](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html) | 검출·인식 단계와 `rec_texts`·`rec_scores`·`rec_polys` 출력 | 채택: 노트북에서 PP-OCRv5를 명시 |
| 8 | 1·3·4 | [PaddleOCR-VL 1.6 소개](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html) | 문서 VLM이 텍스트·표·수식·차트·레이아웃을 함께 다룰 수 있음 | 조건부: 제작사 벤치마크를 절대 성능으로 표현하지 않음 |
| 9 | 3·4·6 | [PaddleOCR-VL 사용법](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) | 영역별 인식 결과를 읽기 순서로 병합하고 JSON·Markdown으로 내보내는 흐름 | 채택 |
| 10 | 4·7 | [JSON Schema 명세](https://json-schema.org/specification) | 자료형·필수 필드·허용 값 등 구조 검증 규칙 | 채택 |
| 11 | 7·8 | [Google Document AI 평가](https://docs.cloud.google.com/document-ai/docs/evaluate) | 정답 라벨과 예측을 비교하는 precision·recall·F1과 임계값의 절충 | 채택 |
| 12 | 1·8 | [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10) | AI 위험을 관리하고 사람 감독의 역할과 책임을 정해야 함 | 채택 |
| 13 | 4·8 | [NIST 생성형 AI 프로파일](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | 생성형 AI의 허구 출력·정보보호·프라이버시 위험을 평가해야 함 | 채택 |
| 14 | 8 | [개인정보보호위원회 생성형 AI 개인정보 처리 안내서 발표](https://pipc.go.kr/np/cop/bbs/selectBoardArticle.do?bbsId=BS074&mCode=C020010000&nttId=11410) | AI 생애주기에서 개인정보 적법성·안전조치와 Privacy by Design을 검토해야 함 | 채택 |
| 15 | 전 교시 | [개인정보보호위원회 AI 이용 개인정보 안내](https://m.pipc.go.kr/np/cop/bbs/selectBoardArticle.do?bbsId=BS212&mCode=C040030000&nttId=12257) | 개인정보 입력 전 확인, 승인된 업무용 도구, 저장·학습 설정 점검 | 채택 |
| 16 | 8·강사용 | [개인정보 보호법 제29조](https://www.law.go.kr/LSW/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1033215737) | 개인정보처리자의 기술적·관리적·물리적 안전조치 의무 | 조건부: 강사용 법적 배경이며 법률 자문으로 표현하지 않음 |
| 17 | 전 교시 | [Google Colab 과거 런타임 버전 FAQ](https://research.google.com/colaboratory/runtime-version-faq.html) | 교육용 노트북의 런타임 고정과 `2026.04`의 Python 3.12.13 확인 | 채택 |
| 18 | 전 교시 | [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) | 무료 Colab 자원과 세션 수명이 동적이며 런타임 파일이 영구 저장되지 않음 | 채택 |
| 19 | 5·6 | [Streamlit 릴리스 노트](https://docs.streamlit.io/develop/quick-reference/release-notes) | 수업 기준 Streamlit 1.60.0 버전 확인 | 채택 |
| 20 | 5·6 | [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) | 브라우저 서버 없이 앱 위젯과 출력을 프로그램으로 검사 | 채택 |
| 21 | 7 | [MITRE CWE-1236](https://cwe.mitre.org/data/definitions/1236.html) | 스프레드시트에서 수식으로 해석될 수 있는 입력값의 무해화 | 채택 |

## 교재 적용 원칙

- 수강생 교재에는 해당 교시의 개념과 행동을 직접 뒷받침하는 자료만 표시합니다.
- 모델 버전 선정 이유, 성능 비교, 가격, 캐시, 라이선스 검증 절차는 강사용 자료에서 관리합니다.
- OCR 신뢰도만으로 자동 승인하지 않습니다. 스키마 검증, 원본 대조, 사람 승인 상태를 별도로 확인합니다.
- 공급업체 벤치마크는 그 공급업체가 보고한 조건의 결과로만 해석합니다.
- 개인정보·법률 자료는 조직 정책과 전문가 검토를 대신하지 않습니다.

## 출처 유지보수

기술 버전·지원 언어·가격·보안 정책은 바뀔 수 있습니다. 강의 전에는 다음 순서로 다시 확인합니다.

1. 공식 문서의 현재 버전과 갱신일을 확인합니다.
2. Colab 노트북과 준비 결과 경로를 새 세션에서 실행합니다.
3. 교재 주장과 제품 문서가 달라졌다면 교재를 먼저 수정합니다.
4. 확인할 수 없는 성능·비용·보안 주장은 교재에서 제거하거나 조건부로 표시합니다.
