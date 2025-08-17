# 슬라이드 36: 하이브리드 Document AI 시스템 실습

**리딩 메시지**: "전통 OCR과 멀티모달 LLM을 효과적으로 결합한 실무용 Document AI 시스템을 구축해보겠습니다"

## 하이브리드 시스템 아키텍처

```
입력 문서 → 품질 평가 → 적응적 라우팅 → 최적 엔진 선택 → 검증 및 보정 → 구조화된 출력
                ↓           ↓                ↓              ↓
              고품질     중품질/복잡         저품질        후처리
                ↓           ↓                ↓              ↓
           전통 OCR    OCR + LLM         LLM 직접       하이브리드 검증
```

## 완전한 하이브리드 Document AI 실습 코드

### 환경 설정
```bash
# 필요한 패키지 설치
pip install opencv-python paddlepaddle paddleocr pytesseract
pip install openai anthropic  # LLM APIs
pip install pillow numpy pandas matplotlib
pip install python-dotenv pydantic  # 구조화 및 검증
pip install fastapi uvicorn  # API 서버 (선택)
```

### 메인 하이브리드 시스템 클래스
```python
import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging

# OCR 엔진들
import pytesseract
from paddleocr import PaddleOCR
import openai
import anthropic

# 이전에 만든 클래스들 import
from multimodal_ocr import MultimodalOCR  # 이전 슬라이드의 코드
from ner_system import UnifiedNERSystem   # 이전 슬라이드의 코드
from preprocessor import DocumentPreprocessor  # 이전 슬라이드의 코드

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    """처리 전략 열거형"""
    TRADITIONAL_OCR = "traditional_ocr"
    HYBRID_OCR_LLM = "hybrid_ocr_llm" 
    MULTIMODAL_LLM = "multimodal_llm"
    ENSEMBLE = "ensemble"

@dataclass
class DocumentQuality:
    """문서 품질 평가 결과"""
    dpi_estimate: float
    blur_score: float
    skew_angle: float
    noise_level: float
    brightness_variance: float
    text_density: float
    complexity_score: float
    recommended_strategy: ProcessingStrategy

class DocumentSchema(BaseModel):
    """문서 스키마 정의"""
    document_type: str = Field(..., description="문서 유형")
    required_fields: Dict[str, str] = Field(..., description="필수 필드들")
    optional_fields: Dict[str, str] = Field(default_factory=dict, description="선택 필드들")
    validation_rules: Dict[str, str] = Field(default_factory=dict, description="검증 규칙들")

class ExtractionResult(BaseModel):
    """추출 결과 모델"""
    success: bool
    processing_time: float
    strategy_used: ProcessingStrategy
    confidence_score: float
    extracted_data: Dict
    raw_text: str
    entities: List[Dict]
    validation_errors: List[str] = Field(default_factory=list)
    cost_estimate: float = 0.0

class HybridDocumentAI:
    """하이브리드 Document AI 시스템"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.initialize_engines()
        self.preprocessor = DocumentPreprocessor()
        self.ner_system = UnifiedNERSystem()
        self.multimodal_ocr = MultimodalOCR()
        
        # 성능 통계
        self.performance_stats = {
            'total_processed': 0,
            'strategy_usage': {strategy: 0 for strategy in ProcessingStrategy},
            'average_processing_time': 0,
            'total_cost': 0
        }
    
    def load_config(self, config_path: Optional[str]) -> Dict:
        """설정 파일 로드"""
        default_config = {
            'quality_thresholds': {
                'high_quality': {'blur_score': 200, 'noise_level': 0.02, 'skew_angle': 1.0},
                'medium_quality': {'blur_score': 100, 'noise_level': 0.05, 'skew_angle': 2.0}
            },
            'cost_limits': {
                'max_llm_tokens_per_doc': 2000,
                'max_cost_per_doc': 0.1
            },
            'confidence_thresholds': {
                'min_ocr_confidence': 0.7,
                'min_llm_confidence': 0.8,
                'ensemble_threshold': 0.9
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def initialize_engines(self):
        """OCR 엔진들 초기화"""
        try:
            # PaddleOCR 초기화
            self.paddleocr = PaddleOCR(use_angle_cls=True, lang='korean', show_log=False)
            logger.info("✓ PaddleOCR 초기화 완료")
        except Exception as e:
            logger.error(f"✗ PaddleOCR 초기화 실패: {e}")
            self.paddleocr = None
        
        # Tesseract 설정
        self.tesseract_config = '--oem 3 --psm 6 -l kor+eng'
        
        logger.info("OCR 엔진 초기화 완료")
    
    def assess_document_quality(self, image_path: str) -> DocumentQuality:
        """문서 품질 종합 평가"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        # 기본 품질 평가 (preprocessor 활용)
        basic_metrics = self.preprocessor.assess_image_quality(image)
        
        # 추가 평가: 텍스트 밀도
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_areas = self.estimate_text_density(gray)
        
        # 복잡도 점수 계산
        complexity_score = self.calculate_complexity_score(image, text_areas)
        
        # 처리 전략 추천
        strategy = self.recommend_processing_strategy(basic_metrics, text_areas, complexity_score)
        
        return DocumentQuality(
            dpi_estimate=basic_metrics['dpi_estimate'],
            blur_score=basic_metrics['blur_score'],
            skew_angle=basic_metrics['skew_angle'],
            noise_level=basic_metrics['noise_level'],
            brightness_variance=basic_metrics['brightness_variance'],
            text_density=text_areas,
            complexity_score=complexity_score,
            recommended_strategy=strategy
        )
    
    def estimate_text_density(self, gray_image: np.ndarray) -> float:
        """텍스트 밀도 추정"""
        # 적응적 이진화
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 연결 요소 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            255 - binary, connectivity=8
        )
        
        # 텍스트로 추정되는 영역 계산
        total_area = gray_image.shape[0] * gray_image.shape[1]
        text_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 텍스트 특성 필터링
            aspect_ratio = width / height if height > 0 else 0
            if 50 < area < 5000 and 0.1 < aspect_ratio < 10:
                text_area += area
        
        return text_area / total_area
    
    def calculate_complexity_score(self, image: np.ndarray, text_density: float) -> float:
        """문서 복잡도 점수 계산"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 에지 밀도
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 2. 히스토그램 복잡도
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # 3. 레이아웃 복잡도 (Hough 라인 개수)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        line_complexity = len(lines) if lines is not None else 0
        
        # 종합 복잡도 점수 (0-1)
        complexity_score = (
            edge_density * 0.3 +
            (hist_entropy / 10000) * 0.3 +
            (line_complexity / 100) * 0.2 +
            text_density * 0.2
        )
        
        return min(complexity_score, 1.0)
    
    def recommend_processing_strategy(self, basic_metrics: Dict, text_density: float, 
                                    complexity_score: float) -> ProcessingStrategy:
        """처리 전략 추천"""
        
        high_quality = self.config['quality_thresholds']['high_quality']
        medium_quality = self.config['quality_thresholds']['medium_quality']
        
        # 고품질 문서: 전통 OCR
        if (basic_metrics['blur_score'] >= high_quality['blur_score'] and
            basic_metrics['noise_level'] <= high_quality['noise_level'] and
            basic_metrics['skew_angle'] <= high_quality['skew_angle'] and
            complexity_score < 0.3):
            return ProcessingStrategy.TRADITIONAL_OCR
        
        # 중품질 문서: 하이브리드
        elif (basic_metrics['blur_score'] >= medium_quality['blur_score'] and
              basic_metrics['noise_level'] <= medium_quality['noise_level'] and
              basic_metrics['skew_angle'] <= medium_quality['skew_angle']):
            return ProcessingStrategy.HYBRID_OCR_LLM
        
        # 복잡하거나 저품질 문서: 멀티모달 LLM
        elif complexity_score > 0.7 or text_density < 0.1:
            return ProcessingStrategy.MULTIMODAL_LLM
        
        # 불확실한 경우: 앙상블
        else:
            return ProcessingStrategy.ENSEMBLE
    
    def process_with_traditional_ocr(self, image_path: str, schema: DocumentSchema) -> Dict:
        """전통 OCR 처리"""
        results = {'method': 'traditional_ocr', 'engines': {}}
        
        # 전처리
        processed_img, binary_img, _ = self.preprocessor.adaptive_preprocessing_pipeline(image_path)
        
        # PaddleOCR
        if self.paddleocr:
            try:
                paddle_results = self.paddleocr.ocr(image_path, cls=True)
                paddle_text = ' '.join([
                    word_info[1][0] for line in paddle_results for word_info in line
                ])
                paddle_confidence = np.mean([
                    word_info[1][1] for line in paddle_results for word_info in line
                ])
                
                results['engines']['paddleocr'] = {
                    'text': paddle_text,
                    'confidence': paddle_confidence
                }
            except Exception as e:
                logger.error(f"PaddleOCR 오류: {e}")
        
        # Tesseract
        try:
            tesseract_text = pytesseract.image_to_string(binary_img, config=self.tesseract_config)
            
            # 신뢰도 계산
            data = pytesseract.image_to_data(binary_img, config=self.tesseract_config, output_type='dict')
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            tesseract_confidence = np.mean(confidences) / 100 if confidences else 0
            
            results['engines']['tesseract'] = {
                'text': tesseract_text.strip(),
                'confidence': tesseract_confidence
            }
        except Exception as e:
            logger.error(f"Tesseract 오류: {e}")
        
        # 최고 신뢰도 엔진 선택
        best_engine = max(results['engines'].items(), 
                         key=lambda x: x[1]['confidence'])[0] if results['engines'] else None
        
        if best_engine:
            results['best_text'] = results['engines'][best_engine]['text']
            results['confidence'] = results['engines'][best_engine]['confidence']
        else:
            results['best_text'] = ""
            results['confidence'] = 0
        
        return results
    
    def process_with_hybrid(self, image_path: str, schema: DocumentSchema) -> Dict:
        """하이브리드 OCR + LLM 처리"""
        # 1. 전통 OCR로 기본 텍스트 추출
        ocr_result = self.process_with_traditional_ocr(image_path, schema)
        
        # 2. LLM으로 구조화 및 보정
        if ocr_result['confidence'] > self.config['confidence_thresholds']['min_ocr_confidence']:
            try:
                # 구조화 프롬프트
                prompt = f"""
다음 OCR로 추출된 텍스트를 분석하고 오류를 수정한 다음, 
주어진 스키마에 맞춰 정보를 추출해주세요.

OCR 텍스트:
{ocr_result['best_text']}

스키마:
{json.dumps(schema.dict(), ensure_ascii=False, indent=2)}

작업:
1. OCR 오류 수정 (잘못 인식된 글자, 누락된 글자 보정)
2. 스키마 필드에 맞는 정보 추출
3. JSON 형식으로 구조화

출력 형식:
{{
  "corrected_text": "오류가 수정된 전체 텍스트",
  "extracted_fields": {{
    "field_name": {{"value": "추출값", "confidence": 0.95}}
  }}
}}
"""
                
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500
                )
                
                llm_content = response.choices[0].message.content
                
                # JSON 파싱
                if '```json' in llm_content:
                    llm_content = llm_content.split('```json')[1].split('```')[0]
                
                llm_result = json.loads(llm_content.strip())
                
                return {
                    'method': 'hybrid',
                    'ocr_result': ocr_result,
                    'llm_result': llm_result,
                    'final_text': llm_result.get('corrected_text', ocr_result['best_text']),
                    'extracted_fields': llm_result.get('extracted_fields', {}),
                    'confidence': min(ocr_result['confidence'] + 0.1, 0.95)  # 보정으로 신뢰도 향상
                }
                
            except Exception as e:
                logger.error(f"LLM 보정 오류: {e}")
                return ocr_result
        
        return ocr_result
    
    def process_with_multimodal_llm(self, image_path: str, schema: DocumentSchema) -> Dict:
        """멀티모달 LLM 직접 처리"""
        try:
            result = self.multimodal_ocr.extract_structured_data(image_path, schema.dict())
            
            return {
                'method': 'multimodal_llm',
                'result': result,
                'confidence': 0.85,  # 멀티모달 LLM 기본 신뢰도
                'cost': 0.003  # 추정 비용
            }
        except Exception as e:
            logger.error(f"멀티모달 LLM 오류: {e}")
            return {'method': 'multimodal_llm', 'error': str(e), 'confidence': 0}
    
    def process_with_ensemble(self, image_path: str, schema: DocumentSchema) -> Dict:
        """앙상블 처리 (모든 방법 결합)"""
        results = {}
        
        # 모든 방법 시도
        results['traditional'] = self.process_with_traditional_ocr(image_path, schema)
        results['hybrid'] = self.process_with_hybrid(image_path, schema)
        results['multimodal'] = self.process_with_multimodal_llm(image_path, schema)
        
        # 신뢰도 기반 최종 선택
        best_method = None
        best_confidence = 0
        
        for method, result in results.items():
            confidence = result.get('confidence', 0)
            if confidence > best_confidence:
                best_confidence = confidence
                best_method = method
        
        if best_method:
            final_result = results[best_method]
            final_result['ensemble_results'] = results
            final_result['selected_method'] = best_method
            return final_result
        
        return {'method': 'ensemble', 'error': '모든 방법 실패', 'confidence': 0}
    
    def validate_extraction(self, extracted_data: Dict, schema: DocumentSchema) -> List[str]:
        """추출 결과 검증"""
        errors = []
        
        # 필수 필드 확인
        for field_name in schema.required_fields.keys():
            if field_name not in extracted_data:
                errors.append(f"필수 필드 누락: {field_name}")
            elif not extracted_data[field_name]:
                errors.append(f"필수 필드 값 없음: {field_name}")
        
        # 검증 규칙 적용
        for field_name, rule in schema.validation_rules.items():
            if field_name in extracted_data:
                value = extracted_data[field_name]
                if not self.apply_validation_rule(value, rule):
                    errors.append(f"검증 실패 - {field_name}: {rule}")
        
        return errors
    
    def apply_validation_rule(self, value: str, rule: str) -> bool:
        """검증 규칙 적용"""
        import re
        
        if rule.startswith('regex:'):
            pattern = rule[6:]  # 'regex:' 제거
            return bool(re.match(pattern, str(value)))
        elif rule.startswith('length:'):
            min_length = int(rule[7:])
            return len(str(value)) >= min_length
        elif rule == 'not_empty':
            return bool(str(value).strip())
        
        return True
    
    def process_document(self, image_path: str, schema: DocumentSchema, 
                        force_strategy: Optional[ProcessingStrategy] = None) -> ExtractionResult:
        """메인 문서 처리 함수"""
        
        start_time = time.time()
        
        try:
            # 1. 문서 품질 평가
            quality = self.assess_document_quality(image_path)
            
            # 2. 처리 전략 결정
            strategy = force_strategy or quality.recommended_strategy
            
            logger.info(f"문서 처리 시작 - 전략: {strategy.value}")
            
            # 3. 전략별 처리
            if strategy == ProcessingStrategy.TRADITIONAL_OCR:
                result = self.process_with_traditional_ocr(image_path, schema)
            elif strategy == ProcessingStrategy.HYBRID_OCR_LLM:
                result = self.process_with_hybrid(image_path, schema)
            elif strategy == ProcessingStrategy.MULTIMODAL_LLM:
                result = self.process_with_multimodal_llm(image_path, schema)
            elif strategy == ProcessingStrategy.ENSEMBLE:
                result = self.process_with_ensemble(image_path, schema)
            else:
                raise ValueError(f"지원하지 않는 전략: {strategy}")
            
            # 4. NER 적용
            text = result.get('final_text') or result.get('best_text') or ""
            entities = self.ner_system.ensemble_ner(text) if text else []
            
            # 5. 추출 데이터 검증
            extracted_data = result.get('extracted_fields', {})
            validation_errors = self.validate_extraction(extracted_data, schema)
            
            # 6. 결과 구성
            processing_time = time.time() - start_time
            
            extraction_result = ExtractionResult(
                success=bool(result.get('confidence', 0) > 0.5),
                processing_time=processing_time,
                strategy_used=strategy,
                confidence_score=result.get('confidence', 0),
                extracted_data=extracted_data,
                raw_text=text,
                entities=entities,
                validation_errors=validation_errors,
                cost_estimate=result.get('cost', 0)
            )
            
            # 7. 통계 업데이트
            self.update_performance_stats(strategy, processing_time, result.get('cost', 0))
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"문서 처리 오류: {e}")
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                success=False,
                processing_time=processing_time,
                strategy_used=strategy if 'strategy' in locals() else ProcessingStrategy.TRADITIONAL_OCR,
                confidence_score=0,
                extracted_data={},
                raw_text="",
                entities=[],
                validation_errors=[f"처리 오류: {str(e)}"]
            )
    
    def update_performance_stats(self, strategy: ProcessingStrategy, 
                                processing_time: float, cost: float):
        """성능 통계 업데이트"""
        self.performance_stats['total_processed'] += 1
        self.performance_stats['strategy_usage'][strategy] += 1
        self.performance_stats['total_cost'] += cost
        
        # 평균 처리 시간 업데이트
        total = self.performance_stats['total_processed']
        current_avg = self.performance_stats['average_processing_time']
        self.performance_stats['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_performance_report(self) -> Dict:
        """성능 리포트 생성"""
        stats = self.performance_stats.copy()
        
        # 전략별 사용 비율 계산
        total = stats['total_processed']
        if total > 0:
            stats['strategy_percentages'] = {
                strategy: (count / total) * 100 
                for strategy, count in stats['strategy_usage'].items()
            }
        
        return stats

# 실행 예제
if __name__ == "__main__":
    # 하이브리드 시스템 초기화
    hybrid_ai = HybridDocumentAI()
    
    # 문서 스키마 정의 (청구서 예시)
    invoice_schema = DocumentSchema(
        document_type="invoice",
        required_fields={
            "invoice_number": "청구서 번호",
            "issue_date": "발행일",
            "company_name": "회사명",
            "total_amount": "총 금액"
        },
        optional_fields={
            "tax_amount": "세금",
            "due_date": "납기일",
            "items": "항목 목록"
        },
        validation_rules={
            "invoice_number": "not_empty",
            "issue_date": "regex:\\d{4}-\\d{2}-\\d{2}",
            "total_amount": "regex:\\d+",
            "company_name": "length:2"
        }
    )
    
    # 문서 처리 테스트
    test_images = [
        "invoice_high_quality.jpg",
        "invoice_scanned.jpg", 
        "invoice_photo.jpg"
    ]
    
    results = []
    
    for image_path in test_images:
        print(f"\n{'='*50}")
        print(f"처리 중: {image_path}")
        
        # 문서 처리
        result = hybrid_ai.process_document(image_path, invoice_schema)
        
        print(f"전략: {result.strategy_used.value}")
        print(f"성공: {result.success}")
        print(f"신뢰도: {result.confidence_score:.2f}")
        print(f"처리 시간: {result.processing_time:.2f}초")
        print(f"비용: ${result.cost_estimate:.4f}")
        
        if result.validation_errors:
            print(f"검증 오류: {result.validation_errors}")
        
        # 추출된 데이터 출력
        print("\n추출된 데이터:")
        for field, value in result.extracted_data.items():
            print(f"  {field}: {value}")
        
        results.append(result)
    
    # 성능 리포트
    print(f"\n{'='*50}")
    print("성능 리포트")
    report = hybrid_ai.get_performance_report()
    
    print(f"총 처리 문서: {report['total_processed']}")
    print(f"평균 처리 시간: {report['average_processing_time']:.2f}초")
    print(f"총 비용: ${report['total_cost']:.4f}")
    
    if 'strategy_percentages' in report:
        print("\n전략별 사용률:")
        for strategy, percentage in report['strategy_percentages'].items():
            print(f"  {strategy}: {percentage:.1f}%")
```

## 성능 최적화 팁

### 1. 배치 처리 최적화
```python
def process_batch(self, image_paths: List[str], schema: DocumentSchema) -> List[ExtractionResult]:
    """배치 문서 처리"""
    results = []
    
    # 품질별 그룹화
    quality_groups = {'high': [], 'medium': [], 'low': []}
    
    for path in image_paths:
        quality = self.assess_document_quality(path)
        if quality.complexity_score < 0.3:
            quality_groups['high'].append(path)
        elif quality.complexity_score < 0.7:
            quality_groups['medium'].append(path)
        else:
            quality_groups['low'].append(path)
    
    # 그룹별 최적 전략으로 처리
    for group, paths in quality_groups.items():
        if group == 'high':
            strategy = ProcessingStrategy.TRADITIONAL_OCR
        elif group == 'medium':
            strategy = ProcessingStrategy.HYBRID_OCR_LLM
        else:
            strategy = ProcessingStrategy.MULTIMODAL_LLM
        
        for path in paths:
            result = self.process_document(path, schema, strategy)
            results.append(result)
    
    return results
```

### 2. 비용 제어
```python
def process_with_cost_control(self, image_path: str, schema: DocumentSchema, 
                             max_cost: float = 0.05) -> ExtractionResult:
    """비용 제한하에 처리"""
    
    # 저비용 전략부터 시도
    strategies = [
        ProcessingStrategy.TRADITIONAL_OCR,
        ProcessingStrategy.HYBRID_OCR_LLM,
        ProcessingStrategy.MULTIMODAL_LLM
    ]
    
    for strategy in strategies:
        estimated_cost = self.estimate_strategy_cost(strategy, image_path)
        if estimated_cost <= max_cost:
            return self.process_document(image_path, schema, strategy)
    
    # 모든 전략이 비용 초과시 가장 저렴한 것 선택
    return self.process_document(image_path, schema, ProcessingStrategy.TRADITIONAL_OCR)
```

---

**강의 섹션**: 8. 엔드투엔드 시스템 구축 (90분)
**슬라이드 번호**: 36/47
