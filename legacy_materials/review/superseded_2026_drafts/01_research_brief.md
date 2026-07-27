# 8교시 Document AI 교재 조사 브리프

조사 기준: 2026-07-27 23:59, Asia/Seoul  
대상: Python 기초 경험이 있는 Document AI 입문자  
기본 환경: Google Colab  
조사 담당: `enterprise_docai_expert`

## 1. 과정 전체 결정

1. Colab 수업 환경을 로컬 Python 패치 버전과 동일하다고 가정하지 않는다. 교재에는 `Python 3.12.x`라고 쓰고 검증에 사용한 Colab 런타임을 별도로 기록한다.
2. 기본 OCR은 API가 단순한 EasyOCR 하나로 제한한다. `Reader(["ko", "en"], gpu=False)`와 `readtext()` 결과의 좌표·텍스트·신뢰도만 다룬다.
3. 모든 실습은 `기본 경로 1개 + mock 경로 1개`만 제공한다. OCR 모델 다운로드 또는 API 호출이 실패하면 제공된 OCR TXT와 JSON으로 넘어간다.
4. Colab에서 생성한 Gradio 공유 주소는 공개 접근이 가능할 수 있다. 실제 개인정보 문서는 사용하지 않고 합성 샘플만 사용한다.
5. JSON Schema 준수와 추출값의 사실성은 별개다. 원문에 없는 값은 `null`로 두고, 원문 근거·형식·업무 규칙·사람 승인을 별도로 확인한다.
6. 엔터프라이즈 내용은 8교시의 개인정보·외부 전송·권한·보존·사람 검토 체크리스트로 집중한다.

## 2. Colab 실행 기준

### 권장 런타임

| 항목 | 검증 기준 |
| --- | --- |
| Colab 런타임 | `2026.04` 고정 런타임 우선 |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.12.13 |
| NumPy | 2.0.2 |
| PyTorch | 2.10.0 |
| 하드웨어 | 기본 CPU, GPU 필수 아님 |

근거: [Google Colab 런타임 버전 FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)

### 최소 설치 후보

```text
easyocr==1.7.2
gradio==6.20.0
PyMuPDF==1.28.0
openai==2.48.0       # 실제 API 선택 실습에서만 설치
```

`torch`, `numpy`, `pandas`, `Pillow`는 Colab 제공 버전을 먼저 사용한다. 첫 셀에서 Python과 핵심 패키지 버전을 출력하고, 깨끗한 `2026.04` CPU 런타임에서 최종 실행 검증을 수행한다.

## 3. 교시별 최소 필수 내용

### 1교시. Document AI 개요

**핵심 3개**

1. OCR은 문자를 읽는 단계이고 Document AI는 OCR, 구조화, 검증, 활용을 포함하는 전체 흐름이다.
2. 과정의 공통 흐름은 `문서 입력 → OCR → 구조화 → 검증 → 저장/활용`이다.
3. 자동화 대상을 정할 때 필요한 필드, 오류 영향, 사람 검토 여부를 먼저 정한다.

**실습**

- 기본: 합성 영수증에서 추출 필드 정의표 작성
- 대체: 영수증의 텍스트 설명만 읽고 같은 표 작성

**제외**

- 클라우드 제품 상세 비교
- 시장점유율, ROI, 생산성 수치
- 모델 학습과 대규모 처리 아키텍처

**공식 근거**

- [Google Cloud Document AI 개요](https://docs.cloud.google.com/document-ai/docs/overview)
- [Amazon Textract 개요](https://docs.aws.amazon.com/textract/latest/dg/what-is.html)
- [Azure AI Document Intelligence 개요](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview?view=doc-intel-4.0.0)

### 2교시. OCR 기본

**핵심 3개**

1. OCR 결과는 좌표, 텍스트, 신뢰도의 묶음으로 볼 수 있다.
2. 해상도, 기울기, 노이즈, 흐림, 잘림은 OCR 결과에 영향을 준다.
3. 신뢰도는 정답 보장이 아니라 검토 순서를 정하는 참고 신호다.

**실습**

- 기본: EasyOCR로 정상·저품질 합성 이미지 비교
- 대체: 준비된 OCR 텍스트와 좌표·신뢰도 JSON 불러오기

**수업용 입력 제한**

- PNG, JPEG, PDF
- 파일당 5MB 이하
- PDF는 최대 3페이지, 200 DPI

이 값은 제품 한계가 아니라 수업 안정성을 위한 정책이라고 명시한다.

**제외**

- OCR 엔진 3종 벤치마크
- 앙상블과 모델 파인튜닝
- 고급 전처리 알고리즘

**공식 근거**

- [EasyOCR Tutorial](https://www.jaided.ai/easyocr/tutorial/)
- [EasyOCR 1.7.2 Release](https://github.com/JaidedAI/EasyOCR/releases/tag/v1.7.2)
- [Google Document AI 지원 파일](https://docs.cloud.google.com/document-ai/docs/file-types)
- [PyMuPDF 이미지 변환](https://pymupdf.readthedocs.io/en/latest/recipes-images.html)

### 3교시. 문서 구조와 정제

**핵심 3개**

1. 키-값, 표, 반복 항목은 서로 다른 구조다.
2. OCR 문자열의 줄 순서가 문서의 논리적 읽기 순서와 항상 같지는 않다.
3. 정제는 공백·줄바꿈·명백한 표기 형식을 정리하는 작업이며 원문에 없는 값을 만드는 작업이 아니다.

**실습**

- 기본: OCR 결과를 헤더·키-값·품목 행으로 재구성
- 대체: 제공된 OCR TXT를 같은 함수에 입력

**안전 원칙**

- 원문과 정제값을 함께 보존한다.
- 어떤 규칙으로 값이 바뀌었는지 확인 가능하게 한다.

**제외**

- 표 선 검출과 셀 병합 알고리즘
- Hough Transform과 Sauvola
- 자동 레이아웃 모델 학습

**공식 근거**

- [Amazon Textract 문서 분석 구조](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-analyzing.html)
- [Amazon Textract 표 구조](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html)
- [Google Document AI 응답 처리](https://docs.cloud.google.com/document-ai/docs/handle-response)

### 4교시. 생성형 AI 정보 추출

**핵심 3개**

1. 추출 전에 필드명, 자료형, 필수 여부를 스키마로 정의한다.
2. 원문에 없는 값은 추측하지 않고 `null`로 반환한다.
3. JSON 문법, 필드 형식, 원문 근거는 서로 다른 검증 단계다.

**실습**

- 기본: OCR 텍스트와 JSON Schema로 추출 프롬프트 작성
- 대체: 같은 스키마의 `extracted_result.json` 불러오기
- 실제 API 호출은 키가 있는 학습자만 수행하는 선택 실습

**안전 원칙**

- API 키는 환경변수 또는 Colab Secrets에서 읽는다.
- 실제 개인정보 문서를 외부 API로 보내지 않는다.
- 공급업체의 보존, 학습 이용, 리전, 계약 조건은 조직 적용 전에 별도 확인한다.

**제외**

- 프롬프트 체인과 Agent 프레임워크
- RAG와 벡터 데이터베이스
- 파인튜닝과 모델 성능 순위

**공식 근거**

- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [OpenAI API 인증](https://platform.openai.com/docs/api-reference/authentication)
- [JSON Schema 2020-12](https://json-schema.org/specification)

### 5교시. Gradio 기본 앱

**핵심 3개**

1. Gradio 컴포넌트는 Python 함수의 입력과 출력을 화면에 연결한다.
2. `File`, `Textbox`, `JSON`, `Dataframe`, `DownloadButton`의 역할만 익힌다.
3. 업로드, 미리보기, 결과 표시를 작은 단위로 만들고 매번 실행한다.

**실습**

- 기본: 합성 이미지 또는 PDF 업로드와 mock 결과 출력
- 대체: 업로드 없이 준비된 샘플 선택

**안전 원칙**

- Colab의 Gradio 공유 주소에는 합성 문서만 올린다.
- 수업 앱은 업로드 크기를 5MB로 제한한다.
- 운영 서비스처럼 사용하지 않는다.

**제외**

- CSS 커스터마이징
- 사용자 계정, 데이터베이스, 배포
- FastAPI 연동

**공식 근거**

- [Gradio File](https://www.gradio.app/docs/gradio/file)
- [Gradio DownloadButton](https://www.gradio.app/docs/gradio/downloadbutton)
- [Gradio 앱 공유](https://www.gradio.app/guides/sharing-your-app)
- [Gradio 파일 접근 보안](https://www.gradio.app/guides/file-access)

### 6교시. OCR·추출 통합

**핵심 3개**

1. 업로드, 파일 변환, OCR, 구조화 추출을 각각 함수로 분리한다.
2. 단계별 상태와 오류 메시지를 보여 준다.
3. 실패 시 mock으로 전환했음을 숨기지 않고 표시한다.

**실습**

- 기본: 업로드 → EasyOCR → mock JSON 추출
- 대체: OCR 또는 파일 변환 예외 시 준비된 TXT와 JSON으로 전환

**안전 원칙**

- 확장자만 믿지 않고 허용 형식, 크기, PDF 페이지 수를 검사한다.
- 원본과 임시 결과는 수업 종료 후 삭제 대상으로 취급한다.

**제외**

- 비동기 작업 큐와 분산 처리
- 데이터베이스 저장과 운영 배포
- 복잡한 재시도 인프라

### 7교시. 검증과 CSV

**핵심 3개**

1. 검증 결과를 `valid`, `warnings`, `errors`로 구분한다.
2. 필수값, 날짜, 금액, 이메일, 전화번호의 형식과 업무 규칙을 확인한다.
3. 전체 정확도 한 숫자보다 필드별 오류와 사람이 수정해야 하는 항목을 본다.

**실습**

- 기본: 정상·누락·잘못된 형식 데이터를 검증하고 UTF-8 BOM CSV 생성
- 대체: 제공된 JSON을 같은 검증·CSV 함수에 전달

**안전 원칙**

- 높은 신뢰도여도 중요 필드는 검토 대상이 될 수 있다.
- 범용 자동 승인 임계값을 사용하지 않는다.
- CSV 셀의 첫 글자가 `=`, `+`, `-`, `@`이면 수식 실행 위험을 피하도록 처리한다.

**제외**

- 복잡한 통계 벤치마크
- 운영 모니터링 대시보드
- Excel 서식 자동화

**공식 근거**

- [Google Document AI 평가](https://docs.cloud.google.com/document-ai/docs/evaluate)
- [OWASP Injection Flaws](https://owasp.org/www-community/Injection_Flaws)

### 8교시. 실무 적용과 사람 검토

**핵심 3개**

1. 자동화 후보는 반복량뿐 아니라 오류 영향과 예외 빈도로 평가한다.
2. 개인정보 최소화, 외부 전송 승인, 접근권한, 보존·삭제를 확인한다.
3. 최종 승인자, 반려·수정 절차, 오류 기록과 개선 책임자를 정한다.

**실습**

- 기본: 자신의 업무를 입력·필드·위험·검토자·저장 형식으로 설계
- 대체: 제공된 견적서 처리 시나리오를 체크리스트로 평가

**제외**

- 규제별 상세 법률 자문
- 대규모 운영 도구와 클라우드 배포
- 모델 재학습과 조직 전체 거버넌스 설계

**공식 근거**

- [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)
- [NIST Generative AI Profile](https://www.nist.gov/itl/ai-risk-management-framework)
- [개인정보보호위원회 생성형 AI 개인정보 처리 안내서](https://pipc.go.kr/np/cop/bbs/selectBoardArticle.do?bbsId=BS074&mCode=C020010000&nttId=11410)
- [개인정보 보호법 제29조](https://www.law.go.kr/LSW/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1033215737)

## 4. 강사가 피해야 할 설명

- “AI가 문서를 사람처럼 이해한다.”
- “JSON Schema를 쓰면 추출한 값도 정확하다.”
- “신뢰도 90%면 자동 승인해도 된다.”
- “전처리를 많이 하면 정확도가 항상 오른다.”
- “Colab 파일은 다음 수업에도 남아 있다.”
- “Gradio 화면이 노트북 안에 보이므로 외부에 공개되지 않는다.”
- “API에 입력한 데이터는 어떤 서비스든 저장되지 않는다.”
- “샘플 하나의 결과가 모든 문서에서의 성능을 대표한다.”

## 5. 기존 자료 판정

### 수정 후 재사용

- OCR과 Document AI의 차이
- `수집 → 판독 → 구조화 → 검증 → 저장` 흐름
- OCR 결과의 텍스트, 신뢰도, 좌표
- 정상·저품질 이미지 비교
- 키-값, 표, 반복 항목
- 원문에 없는 값은 `null`
- 작은 기능 단위의 구현·검토 방식
- 정상·실패·mock 체크리스트

### 폐기

- 구형 PaddleOCR 초기화와 결과 파싱 코드
- `openai.ChatCompletion.create`
- 코드에 API 키 직접 입력
- 세 OCR 엔진 동시 설치와 앙상블
- 범용 `70%`, `90%` 승인 기준
- 출처 없는 정확도, 속도, ROI, 생산성 수치
- 실제 개인정보가 포함된 샘플
- Cursor 또는 Streamlit 전용 안내

## 6. 최종 실행 검증 백로그

### P0

- [ ] 깨끗한 Colab `2026.04` CPU 런타임에서 전체 설치 셀 실행
- [ ] EasyOCR 한국어·영어 모델 최초 다운로드와 실패 경로 확인
- [ ] PNG, JPEG, PDF 입력 확인
- [ ] 손상 PDF, 암호 PDF, 5MB 초과, 3페이지 초과 처리 확인
- [ ] OCR 모델 다운로드 실패 시 mock TXT 전환 확인
- [ ] API 키 없음 또는 API 오류 시 mock JSON 전환 확인
- [ ] Gradio에서 업로드, JSON, 표, CSV 다운로드 확인
- [ ] Gradio 공유 주소에는 합성 문서만 사용
- [ ] 1~8교시 Colab 셀을 새 런타임에서 위에서 아래로 실행

### P1

- [ ] UTF-8 BOM CSV의 한글 Excel 표시 확인
- [ ] OCR 원문, 정제값, 최종값을 구분해 표시
- [ ] mock 전환 상태를 화면에 명확히 표시
- [ ] 실제 API 선택 경로의 현재 SDK 동작 확인
- [ ] 샘플 문서에 실제 개인정보가 없는지 육안 검사

## 7. 기준일 이후 자료 취급

기준일 이후 게시 또는 시행되는 내용은 본문 근거로 사용하지 않는다. 국가법령정보센터에서 개인정보 보호법 제29조의 2026-09-11 시행 예정 내용이 함께 보이더라도 2026-07-27 현재 시행 중인 내용과 구분한다.
