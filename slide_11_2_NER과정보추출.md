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

## 사용자 정의 태깅 구축 방법

### 1. 스키마 정의 (필수 첫 단계)
```json
{
  "INVOICE_NO": "송장번호(영문/숫자/하이픈)",
  "SUPPLIER": "발행사(조직명)",
  "ISSUE_DATE": "발행일(YYYY-MM-DD)",
  "TOTAL_AMT": "총액(숫자, 통화기호 제거)",
  "VAT_AMT": "부가세(숫자)",
  "ADDRESS": "주소(한 줄)"
}
```

### 2. 접근 방식 선택

#### A. 빠른 PoC: LLM 프롬프트 기반
```python
prompt = """
역할: 너는 문서 정보 추출기다. 출력은 반드시 유효한 JSON.

스키마:
- INVOICE_NO: 송장번호(영문/숫자/하이픈)
- SUPPLIER: 발행사(조직명)
- TOTAL_AMT: 총액(숫자, 통화기호 제거)

규칙:
1) 원문에 없는 값은 null
2) 숫자는 쉼표/통화기호 제거
3) 각 필드에 원문 인용 포함

입력: {ocr_text}

출력(JSON):
{{
  "INVOICE_NO": {{"value":"INV-2025-0814","source":"원문"}},
  "SUPPLIER": {{"value":"삼성전자","source":"원문"}},
  "TOTAL_AMT": {{"value": 1234500, "source":"원문"}}
}}
"""
```

#### B. 학습형 NER (높은 정확도)
```python
# 라벨링 도구: Doccano, Label Studio 사용
# 최소 300-2000 문서, 태그당 200-500 스팬 필요

# BIO 태그 예시
tokens = ["홍길동", "이", "2025년", "8월", "14일"]
labels = ["B-PER", "O", "B-DATE", "I-DATE", "I-DATE"]
```

#### C. 규칙 기반 (가성비)
```python
import re

# 정규식 패턴
date_pattern = r'\b(20\d{2})[.\-/년 ](0?[1-9]|1[0-2])[.\-/월 ](0?[1-9]|[12]\d|3[01])\b'
money_pattern = r'\b[₩]?\s?\d{1,3}(,\d{3})*(\.\d+)?\b'

# 앵커 기반 추출
def extract_near_keyword(text, keyword, distance=10):
    # "총액", "합계" 키워드 주변 n자 내 숫자 추출
    pass
```

### 3. 품질 관리 체계

#### 검증 규칙
```python
def validate_extraction(result):
    # 1. JSON 스키마 검증
    # 2. 비즈니스 룰 검증 (합계=항목합?)
    # 3. 신뢰도 임계값 체크
    if result.confidence < 0.8:
        return "manual_review_required"
    return "validated"
```

#### 액티브 러닝 루프
```python
# 1. 사람이 수정한 데이터 수집
# 2. 주기적 재학습 (주/월 단위)  
# 3. 성능 모니터링 및 개선
```

## 완전한 NER 실습 코드

### 환경 설정
```bash
# 필요한 패키지 설치
pip install transformers torch datasets
pip install spacy doccano-client
pip install openai anthropic  # LLM 기반 NER용
pip install pandas matplotlib seaborn
pip install scikit-learn  # 성능 평가용

# spaCy 한국어 모델 설치
python -m spacy download ko_core_news_sm
```

### 통합 NER 시스템 클래스
```python
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Transformers 기반 NER
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# spaCy
import spacy

# LLM 클라이언트  
import openai
import anthropic

# 성능 평가
from sklearn.metrics import classification_report, f1_score

class UnifiedNERSystem:
    def __init__(self):
        self.models = {}
        self.results_history = []
        
        # 사전 정의된 태그 스키마
        self.default_schema = {
            "PER": "사람 이름",
            "ORG": "조직/회사명", 
            "LOC": "위치/주소",
            "DATE": "날짜",
            "MONEY": "금액",
            "PHONE": "전화번호",
            "EMAIL": "이메일",
            "ACCOUNT": "계좌번호",
            "PRODUCT": "제품명",
            "MISC": "기타 중요 정보"
        }
        
        self.load_models()
    
    def load_models(self):
        """다양한 NER 모델 로드"""
        
        # 1. spaCy 한국어 모델
        try:
            self.models['spacy'] = spacy.load('ko_core_news_sm')
            print("✓ spaCy 한국어 모델 로드 완료")
        except:
            print("✗ spaCy 모델 로드 실패")
        
        # 2. Transformers 기반 한국어 NER 모델
        try:
            model_name = "klue/bert-base-ner"
            self.models['transformers_tokenizer'] = AutoTokenizer.from_pretrained(model_name)
            self.models['transformers_model'] = AutoModelForTokenClassification.from_pretrained(model_name)
            self.models['transformers_pipeline'] = pipeline(
                "ner", 
                model=self.models['transformers_model'],
                tokenizer=self.models['transformers_tokenizer'],
                aggregation_strategy="simple"
            )
            print("✓ Transformers 한국어 NER 모델 로드 완료")
        except:
            print("✗ Transformers 모델 로드 실패")
    
    def rule_based_ner(self, text: str) -> List[Dict]:
        """규칙 기반 NER"""
        entities = []
        
        # 정규식 패턴들
        patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE": r'\b(?:\+82-?|0)(?:10|11|16|17|18|19)-?\d{3,4}-?\d{4}\b',
            "DATE": r'\b(20\d{2})[.\-/년\s]+(0?[1-9]|1[0-2])[.\-/월\s]+(0?[1-9]|[12]\d|3[01])[일]?\b',
            "MONEY": r'\b[₩$]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b|\b\d{1,3}(?:,\d{3})*원\b',
            "ACCOUNT": r'\b\d{3}-?\d{2,4}-?\d{6,7}\b',  # 계좌번호 패턴
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'entity': entity_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95,  # 규칙 기반은 높은 신뢰도
                    'method': 'rule_based'
                })
        
        # 키워드 기반 앵커 추출
        anchor_patterns = {
            "ORG": [r'(주식회사|㈜|회사|기업|그룹|코퍼레이션)\s*([가-힣A-Za-z0-9\s]{2,20})', 
                   r'([가-힣A-Za-z0-9\s]{2,20})\s*(주식회사|㈜|회사|기업|그룹)'],
            "PER": [r'(대표|사장|부장|과장|팀장|담당자)[\s:]*([가-힣]{2,4})'],
            "LOC": [r'([가-힣]+[시군구])\s*([가-힣]+[동읍면로길])\s*(\d+[-\d]*)', 
                   r'([가-힣]+도)\s*([가-힣]+[시군])\s*([가-힣]+[구동])']
        }
        
        for entity_type, patterns_list in anchor_patterns.items():
            for pattern in patterns_list:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # 매칭된 그룹에서 실제 엔티티 추출
                    for i, group in enumerate(match.groups()):
                        if group and len(group.strip()) > 1:
                            entities.append({
                                'entity': entity_type,
                                'text': group.strip(),
                                'start': match.start(i+1),
                                'end': match.end(i+1), 
                                'confidence': 0.8,
                                'method': 'anchor_based'
                            })
        
        return entities
    
    def spacy_ner(self, text: str) -> List[Dict]:
        """spaCy 기반 NER"""
        if 'spacy' not in self.models:
            return []
        
        doc = self.models['spacy'](text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'entity': ent.label_,
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.85,  # spaCy는 신뢰도를 제공하지 않으므로 고정값
                'method': 'spacy'
            })
        
        return entities
    
    def transformers_ner(self, text: str) -> List[Dict]:
        """Transformers 기반 NER"""
        if 'transformers_pipeline' not in self.models:
            return []
        
        try:
            results = self.models['transformers_pipeline'](text)
            entities = []
            
            for result in results:
                entities.append({
                    'entity': result['entity_group'],
                    'text': result['word'],
                    'start': result['start'],
                    'end': result['end'],
                    'confidence': result['score'],
                    'method': 'transformers'
                })
            
            return entities
        except Exception as e:
            print(f"Transformers NER 오류: {e}")
            return []
    
    def llm_ner(self, text: str, schema: Optional[Dict] = None, provider: str = "openai") -> List[Dict]:
        """LLM 기반 NER (OpenAI GPT 또는 Anthropic Claude)"""
        
        if schema is None:
            schema = self.default_schema
        
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"""
다음 텍스트에서 명명된 개체(Named Entity)를 추출하고 JSON 형식으로 반환해주세요.

스키마:
{schema_text}

텍스트:
"{text}"

추출 규칙:
1. 위 스키마의 카테고리에 맞는 모든 개체를 찾으세요
2. 각 개체의 시작/끝 위치(character index)를 포함하세요
3. 신뢰도(0-1)를 추정해주세요
4. 중복되거나 겹치는 개체는 제거하세요

출력 형식:
{{
  "entities": [
    {{
      "entity": "PER",
      "text": "홍길동",
      "start": 10,
      "end": 13,
      "confidence": 0.95
    }}
  ]
}}
"""
        
        try:
            if provider == "openai":
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # 비용 효율적인 모델
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
                
            elif provider == "claude":
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-3-haiku-20240307",  # 빠르고 저렴한 모델
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            
            # JSON 파싱
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
                
            result = json.loads(content.strip())
            
            # 결과 표준화
            entities = []
            for ent in result.get('entities', []):
                ent['method'] = f'llm_{provider}'
                entities.append(ent)
            
            return entities
            
        except Exception as e:
            print(f"LLM NER 오류 ({provider}): {e}")
            return []
    
    def ensemble_ner(self, text: str, methods: List[str] = None, schema: Optional[Dict] = None) -> List[Dict]:
        """앙상블 NER - 여러 방법의 결과를 결합"""
        
        if methods is None:
            methods = ['rule_based', 'spacy', 'transformers']
        
        all_entities = []
        
        # 각 방법별로 NER 실행
        for method in methods:
            if method == 'rule_based':
                entities = self.rule_based_ner(text)
            elif method == 'spacy':
                entities = self.spacy_ner(text)
            elif method == 'transformers':
                entities = self.transformers_ner(text)
            elif method.startswith('llm'):
                provider = method.split('_')[1] if '_' in method else 'openai'
                entities = self.llm_ner(text, schema, provider)
            else:
                continue
            
            all_entities.extend(entities)
        
        # 중복 제거 및 신뢰도 기반 선택
        final_entities = self.merge_overlapping_entities(all_entities)
        
        return final_entities
    
    def merge_overlapping_entities(self, entities: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
        """겹치는 엔티티 병합"""
        if not entities:
            return []
        
        # 위치별로 정렬
        entities = sorted(entities, key=lambda x: x['start'])
        merged = []
        
        for entity in entities:
            # 기존 엔티티들과 겹침 검사
            overlapped = False
            
            for i, existing in enumerate(merged):
                # 겹침 정도 계산
                overlap_start = max(entity['start'], existing['start'])
                overlap_end = min(entity['end'], existing['end'])
                overlap_length = max(0, overlap_end - overlap_start)
                
                entity_length = entity['end'] - entity['start']
                existing_length = existing['end'] - existing['start']
                
                overlap_ratio = overlap_length / min(entity_length, existing_length)
                
                if overlap_ratio > overlap_threshold:
                    # 신뢰도가 높은 것으로 교체
                    if entity['confidence'] > existing['confidence']:
                        merged[i] = entity
                    overlapped = True
                    break
            
            if not overlapped:
                merged.append(entity)
        
        return merged
    
    def evaluate_ner_performance(self, test_data: List[Dict]) -> Dict:
        """NER 성능 평가"""
        methods = ['rule_based', 'spacy', 'transformers']
        results = {}
        
        for method in methods:
            y_true = []
            y_pred = []
            
            for item in test_data:
                text = item['text']
                true_entities = item['entities']
                
                # 예측 실행
                if method == 'rule_based':
                    pred_entities = self.rule_based_ner(text)
                elif method == 'spacy':
                    pred_entities = self.spacy_ner(text)
                elif method == 'transformers':
                    pred_entities = self.transformers_ner(text)
                
                # 토큰 레벨 평가를 위한 변환
                true_labels = self.entities_to_bio_labels(text, true_entities)
                pred_labels = self.entities_to_bio_labels(text, pred_entities)
                
                y_true.extend(true_labels)
                y_pred.extend(pred_labels)
            
            # F1 스코어 계산
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            results[method] = {
                'f1_score': f1,
                'classification_report': classification_report(y_true, y_pred, zero_division=0)
            }
        
        return results
    
    def entities_to_bio_labels(self, text: str, entities: List[Dict]) -> List[str]:
        """엔티티를 BIO 태그로 변환"""
        labels = ['O'] * len(text)
        
        for entity in entities:
            start, end = entity['start'], entity['end']
            entity_type = entity['entity']
            
            for i in range(start, min(end, len(text))):
                if i == start:
                    labels[i] = f'B-{entity_type}'
                else:
                    labels[i] = f'I-{entity_type}'
        
        return labels
    
    def save_results(self, text: str, entities: List[Dict], method: str, filename: str = None):
        """결과 저장"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'method': method,
            'entities': entities,
            'entity_count': len(entities)
        }
        
        self.results_history.append(result)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results_history, f, ensure_ascii=False, indent=2)
            print(f"결과가 {filename}에 저장되었습니다.")
    
    def visualize_entities(self, text: str, entities: List[Dict]):
        """엔티티 시각화"""
        print(f"\n원문: {text}")
        print("-" * 50)
        
        # 엔티티별로 그룹화
        entity_groups = {}
        for entity in entities:
            entity_type = entity['entity']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity)
        
        # 엔티티 타입별 출력
        for entity_type, ents in entity_groups.items():
            print(f"\n[{entity_type}]")
            for ent in ents:
                print(f"  - {ent['text']} (신뢰도: {ent['confidence']:.2f}, 방법: {ent['method']})")

# 사용 예제 및 테스트
if __name__ == "__main__":
    ner_system = UnifiedNERSystem()
    
    # 테스트 텍스트
    test_text = """
    삼성전자 주식회사 홍길동 대표이사가 2025년 1월 15일 
    서울특별시 강남구 테헤란로 123에서 열린 기자회견에서 
    신제품 출시를 발표했습니다. 문의사항은 02-1234-5678 또는 
    contact@samsung.com으로 연락주시기 바랍니다.
    계약금 1,500,000원은 110-123-456789 계좌로 입금해주세요.
    """
    
    print("=== 통합 NER 시스템 테스트 ===")
    
    # 1. 개별 방법 테스트
    print("\n1. 규칙 기반 NER:")
    rule_entities = ner_system.rule_based_ner(test_text)
    ner_system.visualize_entities(test_text, rule_entities)
    
    print("\n2. spaCy NER:")
    spacy_entities = ner_system.spacy_ner(test_text)
    ner_system.visualize_entities(test_text, spacy_entities)
    
    print("\n3. Transformers NER:")
    trans_entities = ner_system.transformers_ner(test_text)
    ner_system.visualize_entities(test_text, trans_entities)
    
    # 4. LLM NER (API 키가 있는 경우)
    print("\n4. LLM NER (OpenAI):")
    try:
        llm_entities = ner_system.llm_ner(test_text, provider="openai")
        ner_system.visualize_entities(test_text, llm_entities)
    except Exception as e:
        print(f"LLM NER 스킵 (API 키 필요): {e}")
    
    # 5. 앙상블 NER
    print("\n5. 앙상블 NER:")
    ensemble_entities = ner_system.ensemble_ner(test_text)
    ner_system.visualize_entities(test_text, ensemble_entities)
    
    # 결과 저장
    ner_system.save_results(test_text, ensemble_entities, "ensemble", "ner_results.json")
    
    print("\n테스트 완료!")
```

### 사용자 정의 도메인별 NER 구축 코드
```python
class CustomDomainNER(UnifiedNERSystem):
    """특정 도메인에 특화된 NER 시스템"""
    
    def __init__(self, domain: str = "financial"):
        super().__init__()
        self.domain = domain
        self.load_domain_schema()
    
    def load_domain_schema(self):
        """도메인별 스키마 로드"""
        
        domain_schemas = {
            "financial": {
                "BANK_NAME": "은행명",
                "ACCOUNT_NUM": "계좌번호", 
                "AMOUNT": "금액",
                "TRANSACTION_ID": "거래번호",
                "CARD_NUM": "카드번호",
                "INTEREST_RATE": "이자율",
                "LOAN_TYPE": "대출종류"
            },
            "medical": {
                "DISEASE": "질병명",
                "MEDICINE": "약물명",
                "SYMPTOM": "증상",
                "BODY_PART": "신체부위",
                "MEDICAL_PROCEDURE": "의료절차",
                "DOSAGE": "복용량"
            },
            "legal": {
                "LAW_NAME": "법률명",
                "ARTICLE": "조문",
                "CASE_NUM": "사건번호",
                "COURT": "법원명",
                "JUDGE": "판사명",
                "PLAINTIFF": "원고",
                "DEFENDANT": "피고"
            }
        }
        
        self.domain_schema = domain_schemas.get(self.domain, self.default_schema)
    
    def extract_financial_entities(self, text: str) -> List[Dict]:
        """금융 도메인 특화 추출"""
        entities = []
        
        financial_patterns = {
            "BANK_NAME": r'(KB국민|신한|하나|우리|NH농협|IBK기업)은행',
            "ACCOUNT_NUM": r'\b\d{3}-?\d{2,4}-?\d{6,7}\b',
            "CARD_NUM": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "AMOUNT": r'\b\d{1,3}(?:,\d{3})*원\b|\b[₩$]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',
            "INTEREST_RATE": r'\b\d+(?:\.\d+)?%\b',
            "TRANSACTION_ID": r'\b[A-Z0-9]{8,16}\b'
        }
        
        for entity_type, pattern in financial_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'entity': entity_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9,
                    'method': f'{self.domain}_specialized'
                })
        
        return entities
    
    def process_document(self, text: str, use_llm: bool = True) -> Dict:
        """문서 전체 처리"""
        
        results = {
            'text': text,
            'domain': self.domain,
            'processing_time': 0,
            'entities': {}
        }
        
        start_time = datetime.now()
        
        # 1. 도메인 특화 규칙 적용
        if self.domain == "financial":
            domain_entities = self.extract_financial_entities(text)
        else:
            domain_entities = []
        
        # 2. 기본 NER 방법들 적용
        base_entities = self.ensemble_ner(text, ['rule_based', 'spacy', 'transformers'])
        
        # 3. LLM으로 도메인 특화 정보 추출
        llm_entities = []
        if use_llm:
            try:
                llm_entities = self.llm_ner(text, self.domain_schema)
            except:
                pass
        
        # 4. 결과 통합
        all_entities = domain_entities + base_entities + llm_entities
        final_entities = self.merge_overlapping_entities(all_entities)
        
        # 5. 도메인별 후처리
        final_entities = self.post_process_domain_entities(final_entities)
        
        end_time = datetime.now()
        results['processing_time'] = (end_time - start_time).total_seconds()
        results['entities'] = final_entities
        
        return results
    
    def post_process_domain_entities(self, entities: List[Dict]) -> List[Dict]:
        """도메인별 후처리"""
        
        if self.domain == "financial":
            # 금융 도메인 검증
            for entity in entities:
                if entity['entity'] == 'ACCOUNT_NUM':
                    # 계좌번호 체크섬 검증 (예시)
                    account_num = re.sub(r'[-\s]', '', entity['text'])
                    if len(account_num) < 10:
                        entity['confidence'] *= 0.5  # 신뢰도 하락
                
                elif entity['entity'] == 'AMOUNT':
                    # 금액 정규화
                    amount_text = entity['text']
                    amount_value = re.findall(r'\d{1,3}(?:,\d{3})*', amount_text)
                    if amount_value:
                        entity['normalized_value'] = int(amount_value[0].replace(',', ''))
        
        return entities

# 실행 예제
if __name__ == "__main__":
    # 금융 도메인 NER 테스트
    financial_ner = CustomDomainNER("financial")
    
    financial_text = """
    KB국민은행 계좌번호 110-123-456789로 대출금 5,000,000원을 
    연 3.5% 이자율로 입금하였습니다. 거래번호는 KB20250115001입니다.
    카드번호 1234-5678-9012-3456으로 결제 예정입니다.
    """
    
    result = financial_ner.process_document(financial_text)
    
    print("=== 금융 도메인 NER 결과 ===")
    financial_ner.visualize_entities(financial_text, result['entities'])
    print(f"\n처리 시간: {result['processing_time']:.2f}초")
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
