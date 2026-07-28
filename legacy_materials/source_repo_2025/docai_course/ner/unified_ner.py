"""통합 NER 시스템"""

import re
from typing import List, Dict, Any


class UnifiedNERSystem:
    """규칙 기반 Named Entity Recognition 시스템"""
    
    def __init__(self):
        """NER 패턴 초기화"""
        self.patterns = {
            'date': r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
            'money': r'([0-9,]+)원',
            'phone': r'(\d{2,3}-\d{3,4}-\d{4})',
            'email': r'([\w\.-]+@[\w\.-]+\.\w+)',
            'account': r'(\d{3}-\d{2,6}-\d{2,7})',
        }
    
    def rule_based_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        규칙 기반 정보 추출
        
        Args:
            text: 입력 텍스트
            
        Returns:
            추출된 엔티티 리스트
        """
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'entity': entity_type,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 1.0
                })
        
        return entities
    
    def visualize_entities(self, text: str, entities: List[Dict[str, Any]]):
        """엔티티를 시각화하여 출력"""
        print("=" * 60)
        print("추출된 엔티티:")
        print("=" * 60)
        
        for entity in entities:
            print(f"  [{entity['entity']:10s}] {entity['text']:30s} (신뢰도: {entity['confidence']:.2f})")
        
        print("=" * 60)

