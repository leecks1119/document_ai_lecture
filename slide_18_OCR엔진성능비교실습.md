# 슬라이드 18: OCR 엔진 성능 비교 실습

**리딩 메시지**: "동일 문서로 여러 OCR 엔진을 테스트하여 최적 솔루션을 선택하는 방법을 익히겠습니다"

## 실습 프로세스

1. **테스트 문서 준비**: 계약서, 영수증, 명함, 손글씨 메모 각 10장
2. **OCR 엔진별 테스트**:
   - Tesseract 4.1 (한국어 패키지)
   - EasyOCR (Korean + English)
   - PaddleOCR (multilingual)
   - Google Vision API
3. **성능 지표 측정**:
   - 문자 단위 정확도 (Character Accuracy)
   - 단어 단위 정확도 (Word Accuracy)  
   - 처리 시간 (Processing Time)
   - 비용 (Cost per 1K characters)

## 실습 코드

### 환경 설정
```bash
# 필요한 패키지 설치
pip install pytesseract easyocr paddlepaddle paddleocr opencv-python pillow
pip install google-cloud-vision  # Google Vision API용
pip install Levenshtein  # 정확도 측정용

# Tesseract 한국어 패키지 설치 (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-kor

# Windows의 경우
# https://github.com/UB-Mannheim/tesseract/wiki 에서 다운로드
```

### OCR 엔진 비교 테스트 코드
```python
import cv2
import time
import numpy as np
from PIL import Image
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from google.cloud import vision
import Levenshtein
import json
from datetime import datetime

class OCRBenchmark:
    def __init__(self):
        # OCR 엔진 초기화
        self.tesseract_config = '--oem 3 --psm 6 -l kor+eng'
        self.easyocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)
        self.paddleocr = PaddleOCR(use_angle_cls=True, lang='korean')
        
        # Google Vision API 클라이언트 (사전에 credentials 설정 필요)
        # self.vision_client = vision.ImageAnnotatorClient()
        
        self.results = []

    def preprocess_image(self, image_path):
        """이미지 전처리"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.medianBlur(gray, 3)
        
        # 적응적 이진화
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary, Image.fromarray(binary)

    def tesseract_ocr(self, image_path):
        """Tesseract OCR 테스트"""
        try:
            start_time = time.time()
            processed_img, pil_img = self.preprocess_image(image_path)
            
            # OCR 실행
            text = pytesseract.image_to_string(pil_img, config=self.tesseract_config)
            processing_time = time.time() - start_time
            
            # 신뢰도 정보
            data = pytesseract.image_to_data(pil_img, config=self.tesseract_config, output_type='dict')
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'engine': 'Tesseract',
                'text': text.strip(),
                'processing_time': processing_time,
                'confidence': avg_confidence,
                'cost': 0  # 무료
            }
        except Exception as e:
            return {'engine': 'Tesseract', 'error': str(e)}

    def easyocr_test(self, image_path):
        """EasyOCR 테스트"""
        try:
            start_time = time.time()
            processed_img, _ = self.preprocess_image(image_path)
            
            # OCR 실행
            results = self.easyocr_reader.readtext(processed_img)
            processing_time = time.time() - start_time
            
            # 텍스트 결합
            text = ' '.join([result[1] for result in results])
            
            # 평균 신뢰도
            confidences = [result[2] for result in results]
            avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 0
            
            return {
                'engine': 'EasyOCR',
                'text': text,
                'processing_time': processing_time,
                'confidence': avg_confidence,
                'cost': 0  # 무료
            }
        except Exception as e:
            return {'engine': 'EasyOCR', 'error': str(e)}

    def paddleocr_test(self, image_path):
        """PaddleOCR 테스트"""
        try:
            start_time = time.time()
            
            # OCR 실행
            results = self.paddleocr.ocr(image_path, cls=True)
            processing_time = time.time() - start_time
            
            # 텍스트와 신뢰도 추출
            texts = []
            confidences = []
            
            for line in results:
                for word_info in line:
                    texts.append(word_info[1][0])
                    confidences.append(word_info[1][1])
            
            text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 0
            
            return {
                'engine': 'PaddleOCR',
                'text': text,
                'processing_time': processing_time,
                'confidence': avg_confidence,
                'cost': 0  # 무료
            }
        except Exception as e:
            return {'engine': 'PaddleOCR', 'error': str(e)}

    def google_vision_test(self, image_path):
        """Google Vision API 테스트 (주석 처리 - API 키 필요)"""
        return {
            'engine': 'Google Vision',
            'text': '[API 키 필요 - 실습에서는 생략]',
            'processing_time': 0,
            'confidence': 0,
            'cost': 0.0015  # 추정 비용 (1000자 기준)
        }

    def calculate_accuracy(self, predicted_text, ground_truth):
        """정확도 계산"""
        # 문자 단위 정확도 (Edit Distance 기반)
        char_accuracy = 1 - (Levenshtein.distance(predicted_text, ground_truth) / 
                           max(len(predicted_text), len(ground_truth)))
        
        # 단어 단위 정확도
        pred_words = predicted_text.split()
        truth_words = ground_truth.split()
        
        word_accuracy = 1 - (Levenshtein.distance(' '.join(pred_words), ' '.join(truth_words)) / 
                            max(len(pred_words), len(truth_words)))
        
        return {
            'character_accuracy': max(0, char_accuracy * 100),
            'word_accuracy': max(0, word_accuracy * 100)
        }

    def run_comparison(self, image_path, ground_truth):
        """전체 OCR 엔진 비교 실행"""
        print(f"테스트 이미지: {image_path}")
        print(f"정답 텍스트: {ground_truth}")
        print("-" * 50)
        
        # 각 OCR 엔진 테스트
        engines = [
            self.tesseract_ocr,
            self.easyocr_test,
            self.paddleocr_test,
            self.google_vision_test
        ]
        
        results = []
        for engine_func in engines:
            result = engine_func(image_path)
            
            if 'error' not in result:
                # 정확도 계산
                accuracy = self.calculate_accuracy(result['text'], ground_truth)
                result.update(accuracy)
                
                print(f"엔진: {result['engine']}")
                print(f"추출 텍스트: {result['text'][:100]}...")
                print(f"문자 정확도: {result['character_accuracy']:.2f}%")
                print(f"단어 정확도: {result['word_accuracy']:.2f}%")
                print(f"처리 시간: {result['processing_time']:.2f}초")
                print(f"신뢰도: {result['confidence']:.2f}%")
                print(f"비용: ${result['cost']:.4f}")
                print("-" * 30)
            else:
                print(f"엔진: {result['engine']} - 오류: {result['error']}")
            
            results.append(result)
        
        return results

    def save_results(self, all_results, filename='ocr_benchmark_results.json'):
        """결과를 JSON 파일로 저장"""
        timestamp = datetime.now().isoformat()
        output = {
            'timestamp': timestamp,
            'results': all_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"결과가 {filename}에 저장되었습니다.")

# 실습 실행 예제
if __name__ == "__main__":
    benchmark = OCRBenchmark()
    
    # 테스트 케이스 (실제 사용 시 이미지 경로와 정답 텍스트 입력)
    test_cases = [
        {
            'image': 'test_images/korean_document.jpg',
            'ground_truth': '삼성전자 주식회사 2025년 1월 15일 계약서'
        },
        {
            'image': 'test_images/receipt.jpg', 
            'ground_truth': '편의점 영수증 총액 15,000원'
        }
    ]
    
    all_results = []
    for case in test_cases:
        print(f"\n{'='*60}")
        results = benchmark.run_comparison(case['image'], case['ground_truth'])
        all_results.extend(results)
    
    # 결과 저장
    benchmark.save_results(all_results)
    
    print("\n벤치마크 완료! 결과 파일을 확인하세요.")
```

### 결과 분석 코드
```python
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_ocr_results(results_file='ocr_benchmark_results.json'):
    """OCR 벤치마크 결과 분석"""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    df = pd.DataFrame([r for r in results if 'error' not in r])
    
    # 엔진별 평균 성능
    summary = df.groupby('engine').agg({
        'character_accuracy': 'mean',
        'word_accuracy': 'mean', 
        'processing_time': 'mean',
        'confidence': 'mean',
        'cost': 'mean'
    }).round(2)
    
    print("엔진별 평균 성능:")
    print(summary)
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 정확도 비교
    df.boxplot(column='character_accuracy', by='engine', ax=axes[0,0])
    axes[0,0].set_title('문자 정확도 비교')
    axes[0,0].set_ylabel('정확도 (%)')
    
    # 처리 시간 비교
    df.boxplot(column='processing_time', by='engine', ax=axes[0,1])
    axes[0,1].set_title('처리 시간 비교')
    axes[0,1].set_ylabel('시간 (초)')
    
    # 신뢰도 비교
    df.boxplot(column='confidence', by='engine', ax=axes[1,0])
    axes[1,0].set_title('신뢰도 비교')
    axes[1,0].set_ylabel('신뢰도 (%)')
    
    # 비용 비교
    df.boxplot(column='cost', by='engine', ax=axes[1,1])
    axes[1,1].set_title('비용 비교')
    axes[1,1].set_ylabel('비용 ($)')
    
    plt.tight_layout()
    plt.savefig('ocr_benchmark_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary

# 결과 분석 실행
if __name__ == "__main__":
    summary = analyze_ocr_results()
```

## 실습 진행 방법

1. **코드 복사**: 위 코드를 그대로 복사해서 사용
2. **환경 설정**: 필요한 패키지 설치
3. **테스트 이미지 준비**: 한글이 포함된 다양한 문서 이미지
4. **실행**: 각 OCR 엔진의 성능을 비교 측정
5. **결과 분석**: 정확도, 속도, 비용을 종합적으로 평가

---

**강의 섹션**: 4. 기술요소 실습 (75분)
**슬라이드 번호**: 18/44

