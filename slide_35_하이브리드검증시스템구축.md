# 슬라이드 35: 하이브리드 검증 시스템 구축

**리딩 메시지**: "OCR + LLM + 비즈니스 룰을 결합한 다층 검증으로 99.5% 정확도를 달성합니다"

## 3단계 검증 아키텍처

### 1차: OCR 신뢰도 기반 필터링
```python
def primary_filter(ocr_result):
    high_confidence = []
    low_confidence = []
    
    for item in ocr_result:
        if item.confidence > 0.9:
            high_confidence.append(item)
        else:
            low_confidence.append(item)
    
    return high_confidence, low_confidence
```

### 2차: LLM 기반 문맥 보정
```python
def llm_correction(low_confidence_items, context):
    prompt = f"""
    다음 OCR 결과를 문맥을 고려하여 수정하세요:
    문맥: {context}
    OCR 결과: {low_confidence_items}
    """
    return llm.generate(prompt)
```

### 3차: 비즈니스 룰 검증
```python
def business_rule_validation(text, document_type):
    if document_type == "계약서":
        validate_contract_format(text)
        validate_legal_terms(text)
        validate_date_consistency(text)
    elif document_type == "영수증":
        validate_receipt_format(text)
        validate_tax_calculation(text)
    
    return validation_result
```

---

**강의 섹션**: 7. GPT급 LLM을 통한 OCR 보정 (90분)
**슬라이드 번호**: 35/44
