# 과거 자료와 새 8교시 매핑

## 1교시. Document AI 개요 및 문서 자동화 활용 사례

### 참고 자료

- `00_course_home.md`
- `02_document_ai_overview.md`
- `12_business_ideas.md`
- `13_2026_renewal_plan.md`

### 활용 내용

- 비정형 문서를 JSON·CSV 같은 구조화 데이터로 변환한다는 정의
- OCR와 Document AI의 차이
- 문서 수집→판독→구조화→검증→저장 흐름
- 영수증, 신청서, 견적서, 점검표 사례
- 문서 유형, 필드, 데이터 형식, 필수 여부, 실패 영향을 정의하는 실습

### 새로 보완할 내용

- “OCR은 읽기, Document AI는 서류를 장부에 옮기는 업무” 비유
- 생성형 AI와 전문 문서 모델의 역할 구분
- 사람 검토와 개인정보 원칙
- 2026-07-27 공식 출처

## 2교시. OCR 기반 텍스트 추출

### 참고 자료

- `03_paddleocr_basic.md`
- `04_ocr_engine_comparison.md`
- `05_confidence_visualization.md`
- `06_image_preprocessing.md`

### 활용 내용

- OCR 입력과 출력
- 텍스트·신뢰도·바운딩 박스
- 정상 문서와 저품질 문서 비교
- Ground Truth와 오류 찾기
- 노이즈와 기울기가 인식에 미치는 영향

### 단순화

- PaddleOCR 또는 EasyOCR 하나만 사용
- 세 엔진 벤치마크 제외
- 설치 실패 시 `sample_outputs/ocr_result.txt` 사용

## 3교시. 문서 구조 이해 및 결과 정제

### 참고 자료

- `06_image_preprocessing.md`
- `08_table_extraction.md`
- `09_regex_information_extraction.md`

### 활용 내용

- 키-값, 표, 반복 항목 구조
- 읽기 순서와 위치 관계
- 행·열과 헤더 개념
- 공백, 줄바꿈, 숫자 오인식 정리

### 단순화

- 표 검출 알고리즘 구현 제외
- 제공된 OCR 결과를 항목 또는 행 단위로 직접 정리

## 4교시. 생성형 AI 기반 핵심 정보 추출

### 참고 자료

- `09_regex_information_extraction.md`
- `02_document_ai_overview.md`

### 활용 내용

- 영수증 필드와 JSON 스키마
- 원문에 없는 값은 `null`
- 날짜·금액 정규화
- OCR 텍스트→프롬프트→JSON→검증 흐름

### 새로 보완할 내용

- 추출 근거를 원문에서 확인하는 절차
- 실제 API 없이 가능한 mock JSON
- 최신 구조화 출력 방식
- API 키를 `.env`로 관리

## 5교시. Gradio 기반 앱 기본 구현

### 참고 자료

- `10_cursor_toy_project.md`
- `01_development_environment.md`

### 활용 내용

- 요구사항→코드 생성→검토→실행→수정 반복
- 파일 업로드, 이미지 미리보기, JSON과 표 출력
- 기능을 작은 단위로 나누어 요청하는 방법

### 변경

- Cursor를 Codex로 교체
- 저장소의 `src/` 구조와 연결
- Gradio를 기본 UI로 사용

## 6교시. OCR와 정보 추출 통합

### 참고 자료

- `03_paddleocr_basic.md`
- `09_regex_information_extraction.md`
- `10_cursor_toy_project.md`

### 활용 내용

- OCR 모듈과 추출 모듈 분리
- 업로드→OCR→JSON 연결
- 예외 처리와 오류 메시지

### 새로 보완할 내용

- 실제 OCR와 mock OCR 전환
- 실제 AI 호출과 mock 추출 전환
- 설치 실패가 앱 전체 실패로 이어지지 않는 구조

## 7교시. 검증 및 데이터 저장

### 참고 자료

- `05_confidence_visualization.md`
- `08_table_extraction.md`
- `09_regex_information_extraction.md`
- `11_system_test_qna.md`

### 활용 내용

- 필수값 누락
- 날짜·금액·이메일·전화번호 형식
- 오류와 경고 구분
- JSON→DataFrame→CSV
- 정상·오류 데이터 테스트

### 변경

- OCR 평균 신뢰도만으로 승인하지 않음
- 필드 중요도와 검증 결과를 함께 사용

## 8교시. 실무 적용과 최종 정리

### 참고 자료

- `00_course_home.md`
- `11_system_test_qna.md`
- `12_business_ideas.md`

### 활용 내용

- 정상·저품질·누락·mock 흐름 최종 점검
- 업무 자동화 후보 선택
- 입력 문서, 추출 필드, 검토자, 저장 형태 설계
- 보안·개인정보·실패 영향 검토

### 제외

- 프로파일링, 운영 로그 시스템, Docker, 실제 외부 서비스 연동

## 연결 요약

```text
기존 OCR 중심 11개 교재
    ↓ 개념과 실습 아이디어 선별
새 1~4교시: 이해·OCR·구조화·JSON
    ↓ 프로젝트 기능으로 재구성
새 5~7교시: Gradio·통합·검증·CSV
    ↓ 운영 관점 추가
새 8교시: 사람 검토와 업무 적용
```
