# 슬라이드 11-2: NER과 정보 추출

**리딩 메시지**: "단순한 텍스트 추출을 넘어 의미 있는 정보를 자동으로 식별하고 구조화합니다"

## NER(Named Entity Recognition)의 중요성

### 정의와 목적
- **NER**: 텍스트에서 인명, 지명, 조직명, 날짜, 금액 등 특정 개체를 식별
- **Document AI에서의 역할**: OCR 후 추출된 텍스트에서 핵심 정보만 선별
- **비즈니스 가치**: 수작업 검토 시간 80% 단축

### 주요 엔티티 유형

#### 범용 엔티티
- **PERSON**: 인명 (홍길동, John Smith)
- **ORG**: 조직명 (삼성전자, Google)
- **GPE**: 지정학적 위치 (서울시, 미국)
- **DATE**: 날짜 (2025년 1월 15일)
- **MONEY**: 금액 (1,000,000원, $500)

#### 도메인 특화 엔티티
- **금융**: 계좌번호, 카드번호, 금리
- **의료**: 질병명, 약물명, 의료기관
- **법무**: 계약조건, 법조항, 당사자
- **제조**: 제품코드, 규격, 품질기준

## 최신 NER 기술 (2025년)

### 전통적 방법
- **규칙 기반**: 정규표현식, 사전 매칭
- **통계 모델**: CRF, SVM 기반 시퀀스 라벨링

### 딥러닝 기반 (현재 주류)
- **BERT 계열**: KoBERT, RoBERTa 등 한국어 특화 모델
- **spaCy**: 다국어 지원, 빠른 처리 속도
- **Flair**: 문자 수준 임베딩으로 높은 정확도

### LLM 기반 (2025년 최신)
- **GPT-4**: 제로샷 엔티티 추출, 복잡한 관계 파악
- **Claude 3**: 구조화된 출력, JSON 형태 결과 제공
- **한국어 특화**: KoGPT, HyperCLOVA 등

## 실무 적용 사례

### 계약서 처리
```python
def extract_contract_entities(text):
    entities = {
        "parties": extract_parties(text),      # 계약 당사자
        "amounts": extract_money(text),        # 계약 금액
        "dates": extract_dates(text),          # 계약 기간
        "clauses": extract_clauses(text)       # 주요 조항
    }
    return entities
```

### 영수증 처리
```python
def extract_receipt_entities(text):
    return {
        "merchant": extract_organization(text),  # 가맹점명
        "total_amount": extract_total(text),     # 총액
        "date": extract_transaction_date(text),  # 거래일시
        "items": extract_line_items(text)        # 상품 목록
    }
```

## 성능 향상 팁

### 1. 도메인 적응
- 업계별 전문용어 사전 구축
- 도메인 특화 모델 파인튜닝
- 규칙과 ML 모델 하이브리드 접근

### 2. 품질 개선
- 다중 모델 앙상블
- 신뢰도 기반 후처리
- 사용자 피드백 학습

### 3. 처리 최적화
- 배치 처리로 처리량 향상
- 캐싱으로 중복 계산 방지
- GPU 가속 활용

---

**강의 섹션**: 2. Document AI 기술요소 (60분)
**슬라이드 번호**: 11-2/44
