# 기존 교재 11. 전체 시스템 테스트와 Q&A

- 원본: [전체 시스템 테스트, QnA](https://app.notion.com/p/282707c7ae758117ba44f45cb6feee02)
- Notion 조회 시점의 최종 수정일: 2025-10-04

## 학습 목표

- 엔드투엔드 시스템 검증
- 성능과 병목 분석
- 배포·운영 고려사항 확인
- 오류 대응

## 코드 자산

### 통합 테스트

- `unittest` 기반 영수증과 명함 처리 테스트
- 배치 처리 결과의 성공률 검사

### 성능 측정

- PaddleOCR와 EasyOCR 처리 시간 비교
- 전처리 단계 총 시간 측정
- pandas와 matplotlib 결과 표·차트

### 정확도 평가

- Ground Truth CSV
- `difflib.SequenceMatcher` 기반 유사도
- 정확도 리포트 파일 생성

### 병목 분석

- `cProfile`과 `pstats`

### 배포 점검

- 패키지 설치 여부
- OCR 모델 로딩
- 디스크 여유 공간

### 운영 모니터링

- 처리 건수, 성공·실패, 처리 시간 기록
- JSON Lines 형태의 운영 로그
- 일일 통계 출력

## FAQ

- OCR 정확도가 낮을 때: 입력 품질, 전처리, 도메인 모델 검토
- 속도가 느릴 때: 이미지 크기, GPU, 배치 처리 검토
- 특정 문서 유형이 약할 때: 문서별 규칙과 샘플 개선
- 메모리 부족: 해상도와 배치 크기 조정

## 문제점

- `UnifiedDocumentProcessor`, `RobustDocumentProcessor`, `PreprocessingPipeline` 등 정의되지 않은 클래스에 의존
- 80% 성공률과 90% 문자열 유사도 기준에 근거가 없음
- Python 패키지명과 import 이름이 달라 의존성 검사 코드가 일부 실패할 수 있음
- 배포·프로파일링·운영 모니터링은 8시간 입문 과정에 과함

## 개편 판단

- 정상·오류·mock 흐름 테스트는 8교시 최종 점검으로 재사용
- 필수값과 형식 검증 단위 테스트는 7교시로 이동
- 성능 프로파일링과 운영 모니터링은 제외
- FAQ는 `docs/troubleshooting.md`의 초안 자료로 활용

판정: **테스트 시나리오와 FAQ만 재사용**
