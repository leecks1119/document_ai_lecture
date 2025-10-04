"""OCR 엔진 성능 비교 도구"""

import time
import json
from typing import Dict, List, Any
import Levenshtein


class OCRBenchmark:
    """여러 OCR 엔진의 성능을 비교하는 클래스"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Args:
            use_gpu: GPU 사용 여부
        """
        self.use_gpu = use_gpu
        self.engines = {}
        
    def run_comparison(self, image_path: str, ground_truth: str) -> List[Dict[str, Any]]:
        """
        여러 OCR 엔진으로 이미지를 처리하고 결과를 비교
        
        Args:
            image_path: 이미지 파일 경로
            ground_truth: 정답 텍스트
            
        Returns:
            각 엔진별 결과 리스트
        """
        import pytesseract
        from PIL import Image
        
        results = []
        
        # Tesseract
        try:
            start = time.time()
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='kor+eng')
            end = time.time()
            
            acc = self._calculate_accuracy(text, ground_truth)
            
            results.append({
                'engine': 'Tesseract',
                'text': text.strip(),
                'character_accuracy': acc,
                'processing_time': end - start,
                'confidence': 0.0
            })
        except Exception as e:
            results.append({'engine': 'Tesseract', 'error': str(e)})
        
        # PaddleOCR
        try:
            from paddleocr import PaddleOCR
            start = time.time()
            ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=self.use_gpu)
            result = ocr.ocr(image_path, cls=True)
            end = time.time()
            
            if result and result[0]:
                text = ' '.join([line[1][0] for line in result[0]])
                conf = sum([line[1][1] for line in result[0]]) / len(result[0])
                acc = self._calculate_accuracy(text, ground_truth)
                
                results.append({
                    'engine': 'PaddleOCR',
                    'text': text,
                    'character_accuracy': acc,
                    'processing_time': end - start,
                    'confidence': conf
                })
        except Exception as e:
            results.append({'engine': 'PaddleOCR', 'error': str(e)})
        
        # EasyOCR
        try:
            import easyocr
            start = time.time()
            reader = easyocr.Reader(['ko', 'en'], gpu=self.use_gpu)
            result = reader.readtext(image_path)
            end = time.time()
            
            if result:
                text = ' '.join([item[1] for item in result])
                conf = sum([item[2] for item in result]) / len(result)
                acc = self._calculate_accuracy(text, ground_truth)
                
                results.append({
                    'engine': 'EasyOCR',
                    'text': text,
                    'character_accuracy': acc,
                    'processing_time': end - start,
                    'confidence': conf
                })
        except Exception as e:
            results.append({'engine': 'EasyOCR', 'error': str(e)})
        
        return results
    
    def _calculate_accuracy(self, predicted: str, ground_truth: str) -> float:
        """문자 단위 정확도 계산"""
        distance = Levenshtein.distance(predicted, ground_truth)
        max_len = max(len(predicted), len(ground_truth))
        if max_len == 0:
            return 0.0
        return max(0, (1 - distance / max_len) * 100)
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """결과를 JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

