# 기존 교재 10. Cursor AI 토이 프로젝트

- 원본: [Cursor AI 활용 토이 프로젝트](https://app.notion.com/p/282707c7ae7581f48a7ec6f14c949548)
- Notion 조회 시점의 최종 수정일: 2025-10-04

## 학습 목표

- AI 페어 프로그래밍 경험
- Document AI 시스템 구축
- OCR와 정보 추출 기능 연결
- 코드 리팩터링과 테스트

## AI 코딩 워크플로

요구사항 정의 → AI에 요청 → 코드 생성 → 코드 검토 → 실행 테스트 → 오류 피드백 → 리팩터링 → 완성

이 반복 흐름은 도구를 Codex로 바꾸어도 그대로 활용할 수 있다.

## 기존 도구 설명

- Cursor 설치
- Cmd+K 코드 생성·수정
- Cmd+L 채팅
- 다중 파일 참조
- 오류 메시지 전달
- 코드 설명, 주석, pytest 생성 요청

## 프로젝트 1: 영수증 처리

### 요구사항

- OCR 텍스트 추출
- 정규식과 LLM 기반 핵심 정보 추출
- JSON 출력
- 웹 인터페이스

### 제안된 구조

```text
receipt_processor/
├── main.py
├── ocr_module.py
├── extractor.py
├── llm_corrector.py
├── web_app.py
├── requirements.txt
└── README.md
```

### 포함된 프롬프트

- PaddleOCR 클래스 생성
- 이미지 전처리, 신뢰도, 예외 처리
- 영수증 필드 추출
- Gradio 파일 업로드, JSON, 표 결과 화면

### 코드 자산

- `ReceiptOCR`
- 그레이스케일, 노이즈 제거, CLAHE, Otsu 이진화
- OCR 결과 텍스트와 평균 신뢰도 반환
- 로깅과 타입 힌트

## 프로젝트 2: 명함 분석

- 이름, 회사, 전화, 이메일 추출
- 중복 확인
- CSV 내보내기

## 완성 체크리스트

- OCR, 정보 추출, LLM 보정, 웹 UI, 예외 처리
- 로깅, 단위 테스트, 문서화, 타입 힌트, 설정 분리
- `requirements.txt`, README, `.gitignore`, 선택적 Docker

## 문제점

- Cursor 설치·단축키·API 키 안내가 새 Codex 과정과 맞지 않음
- “생산성 3~5배” 주장은 근거가 없음
- 90분 분량이며 데이터베이스와 Docker까지 포함해 범위가 과함
- PaddleOCR API가 다른 페이지와 혼재
- AI가 생성한 코드를 검증하는 기준보다 생성 기능 소개 비중이 큼

## 개편 판단

- 반복적인 AI 코딩 워크플로는 5교시에 Codex 방식으로 재사용
- Gradio 요청 프롬프트는 새 앱 구조에 맞게 수정
- 프로젝트 구조는 새 `src/` 모듈 구성으로 통합
- 영수증 처리 흐름은 5~7교시 전체 프로젝트의 기반으로 활용
- 명함, DB, Docker는 제외

판정: **프로젝트 흐름과 프롬프트는 재사용, 도구 설명과 코드는 재작성**
