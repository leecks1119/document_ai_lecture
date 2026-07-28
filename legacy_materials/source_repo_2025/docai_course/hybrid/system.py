"""하이브리드 Document AI 시스템"""

from typing import Dict, Any


class HybridDocumentAI:
    """전처리, OCR, NER을 통합한 하이브리드 시스템"""
    
    def __init__(self, use_preprocessing: bool = True, use_ensemble: bool = False, use_llm: bool = False):
        """
        Args:
            use_preprocessing: 전처리 사용 여부
            use_ensemble: OCR 앙상블 사용 여부
            use_llm: LLM 보정 사용 여부
        """
        self.use_preprocessing = use_preprocessing
        self.use_ensemble = use_ensemble
        self.use_llm = use_llm
        
        if self.use_preprocessing:
            from docai_course.preprocessing import DocumentPreprocessor
            self.preprocessor = DocumentPreprocessor()
        
        if self.use_ensemble:
            from docai_course.ocr import OCRBenchmark
            self.ocr = OCRBenchmark()
        
    def process(self, image_path: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        이미지를 처리하여 결과 반환
        
        Args:
            image_path: 이미지 파일 경로
            ground_truth: 정답 텍스트 (평가용, 선택사항)
            
        Returns:
            처리 결과 딕셔너리
        """
        results = {
            'image_path': image_path,
            'preprocessing': None,
            'ocr': None,
            'ner': None
        }
        
        # 전처리
        if self.use_preprocessing:
            processed, binary, metrics = self.preprocessor.adaptive_preprocessing_pipeline(image_path)
            results['preprocessing'] = {
                'quality_metrics': metrics,
                'processed': True
            }
            # 전처리된 이미지로 OCR 수행
            import cv2
            temp_path = 'temp_preprocessed.jpg'
            cv2.imwrite(temp_path, binary)
            image_path = temp_path
        
        # OCR
        if self.use_ensemble and ground_truth:
            ocr_results = self.ocr.run_comparison(image_path, ground_truth)
            results['ocr'] = ocr_results
        
        return results

