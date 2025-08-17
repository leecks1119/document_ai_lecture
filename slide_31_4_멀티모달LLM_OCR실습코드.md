# 슬라이드 31-4: 멀티모달 LLM OCR 실습 코드

**리딩 메시지**: "깃허브에서 바로 복사해서 사용할 수 있는 멀티모달 LLM OCR 실습 코드입니다"

## GPT-4V 기반 OCR 실습

### 환경 설정
```bash
# 필요한 패키지 설치
pip install openai pillow base64 requests python-dotenv
pip install opencv-python paddleocr  # 비교용 전통 OCR
```

### OpenAI GPT-4V OCR 클래스
```python
import base64
import requests
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import json
import time
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class MultimodalOCR:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. .env 파일에 OPENAI_API_KEY를 설정하세요.")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def encode_image_to_base64(self, image_path):
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def preprocess_image_for_llm(self, image_path, max_size=1024):
        """LLM 처리를 위한 이미지 전처리"""
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        # 크기 조정 (토큰 비용 절약을 위해)
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 대비 향상 (LLM이 텍스트를 더 잘 인식하도록)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 임시 파일로 저장
        temp_path = "temp_enhanced.jpg"
        cv2.imwrite(temp_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return temp_path
    
    def extract_text_with_coordinates(self, image_path, detail_level="high"):
        """좌표 정보와 함께 텍스트 추출"""
        
        # 이미지 전처리
        processed_path = self.preprocess_image_for_llm(image_path)
        base64_image = self.encode_image_to_base64(processed_path)
        
        prompt = """
이 이미지에서 모든 텍스트를 추출하고 다음 JSON 형식으로 반환해주세요:

{
  "extracted_text": "전체 텍스트 내용",
  "text_blocks": [
    {
      "text": "개별 텍스트 블록",
      "type": "title|body|table|header|footer",
      "confidence": 0.95,
      "language": "korean|english|mixed"
    }
  ],
  "document_info": {
    "document_type": "invoice|contract|receipt|letter|form",
    "language": "korean|english|mixed",
    "layout": "single_column|multi_column|table|mixed"
  }
}

규칙:
1. 모든 한글, 영문, 숫자를 정확히 추출
2. 표나 양식의 구조를 유지
3. 읽을 수 없는 부분은 [UNCLEAR]로 표시
4. JSON 형식을 정확히 지켜주세요
"""
        
        payload = {
            "model": "gpt-4o",  # 또는 "gpt-4-vision-preview"
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": detail_level
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1  # 일관된 결과를 위해 낮은 온도
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # JSON 파싱 시도
            try:
                # 코드 블록 제거 (```json...``` 형태)
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                
                parsed_result = json.loads(content.strip())
                
                # 처리 시간과 비용 정보 추가
                usage = result.get('usage', {})
                parsed_result['processing_info'] = {
                    'tokens_used': usage.get('total_tokens', 0),
                    'estimated_cost': usage.get('total_tokens', 0) * 0.00003,  # GPT-4V 추정 비용
                    'model': payload['model']
                }
                
                return parsed_result
                
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 원본 텍스트 반환
                return {
                    "extracted_text": content,
                    "error": "JSON 파싱 실패",
                    "raw_response": content
                }
        
        except requests.exceptions.RequestException as e:
            return {"error": f"API 요청 실패: {str(e)}"}
        
        finally:
            # 임시 파일 정리
            if os.path.exists(processed_path):
                os.remove(processed_path)
    
    def extract_structured_data(self, image_path, schema):
        """사용자 정의 스키마에 따른 구조화된 데이터 추출"""
        
        processed_path = self.preprocess_image_for_llm(image_path)
        base64_image = self.encode_image_to_base64(processed_path)
        
        # 스키마를 프롬프트에 포함
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"""
이 문서 이미지에서 다음 스키마에 맞춰 정보를 추출해주세요:

{schema_text}

규칙:
1. 스키마의 모든 필드를 채우려고 시도하세요
2. 문서에 없는 정보는 null로 설정
3. 날짜는 YYYY-MM-DD 형식으로 변환
4. 숫자에서 통화 기호와 쉼표 제거
5. 각 필드에 대해 원문에서 찾은 텍스트를 source에 포함
6. 정확한 JSON 형식으로 응답

예시 출력:
{{
  "field_name": {{
    "value": "추출된 값",
    "source": "원문 텍스트",
    "confidence": 0.95
  }}
}}
"""
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # JSON 파싱
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            return json.loads(content.strip())
            
        except Exception as e:
            return {"error": f"구조화된 데이터 추출 실패: {str(e)}"}
        
        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)
    
    def compare_with_traditional_ocr(self, image_path):
        """전통적인 OCR과 성능 비교"""
        from paddleocr import PaddleOCR
        import pytesseract
        
        # 전통적인 OCR 실행
        paddleocr = PaddleOCR(use_angle_cls=True, lang='korean')
        
        # PaddleOCR
        start_time = time.time()
        paddle_results = paddleocr.ocr(image_path, cls=True)
        paddle_time = time.time() - start_time
        
        paddle_text = ' '.join([
            word_info[1][0] for line in paddle_results for word_info in line
        ])
        
        # Tesseract
        image = cv2.imread(image_path)
        start_time = time.time()
        tesseract_text = pytesseract.image_to_string(image, lang='kor+eng')
        tesseract_time = time.time() - start_time
        
        # 멀티모달 LLM
        start_time = time.time()
        llm_result = self.extract_text_with_coordinates(image_path)
        llm_time = time.time() - start_time
        
        comparison = {
            "traditional_ocr": {
                "paddleocr": {
                    "text": paddle_text.strip(),
                    "processing_time": paddle_time,
                    "cost": 0  # 무료
                },
                "tesseract": {
                    "text": tesseract_text.strip(),
                    "processing_time": tesseract_time,
                    "cost": 0  # 무료
                }
            },
            "multimodal_llm": {
                "text": llm_result.get('extracted_text', ''),
                "processing_time": llm_time,
                "cost": llm_result.get('processing_info', {}).get('estimated_cost', 0),
                "tokens_used": llm_result.get('processing_info', {}).get('tokens_used', 0),
                "structured_output": llm_result
            }
        }
        
        return comparison

# 사용 예제
if __name__ == "__main__":
    # API 키 설정 (.env 파일에 OPENAI_API_KEY=your_key_here)
    ocr = MultimodalOCR()
    
    # 1. 기본 텍스트 추출
    print("=== 기본 텍스트 추출 ===")
    image_path = "sample_document.jpg"
    
    try:
        result = ocr.extract_text_with_coordinates(image_path)
        print(f"추출된 텍스트: {result.get('extracted_text', '')[:200]}...")
        print(f"사용된 토큰: {result.get('processing_info', {}).get('tokens_used', 0)}")
        print(f"예상 비용: ${result.get('processing_info', {}).get('estimated_cost', 0):.4f}")
    except Exception as e:
        print(f"오류: {e}")
    
    # 2. 구조화된 데이터 추출 (청구서 예시)
    print("\n=== 구조화된 데이터 추출 ===")
    invoice_schema = {
        "invoice_number": "청구서 번호",
        "issue_date": "발행일 (YYYY-MM-DD)",
        "company_name": "회사명",
        "total_amount": "총 금액 (숫자만)",
        "tax_amount": "세금 (숫자만)",
        "items": [
            {
                "description": "상품/서비스 설명",
                "quantity": "수량",
                "unit_price": "단가",
                "amount": "금액"
            }
        ]
    }
    
    try:
        structured_result = ocr.extract_structured_data(image_path, invoice_schema)
        print(json.dumps(structured_result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"오류: {e}")
    
    # 3. 전통적인 OCR과 비교
    print("\n=== OCR 엔진 비교 ===")
    try:
        comparison = ocr.compare_with_traditional_ocr(image_path)
        
        print("PaddleOCR:")
        print(f"  텍스트: {comparison['traditional_ocr']['paddleocr']['text'][:100]}...")
        print(f"  처리시간: {comparison['traditional_ocr']['paddleocr']['processing_time']:.2f}초")
        
        print("\nTesseract:")
        print(f"  텍스트: {comparison['traditional_ocr']['tesseract']['text'][:100]}...")
        print(f"  처리시간: {comparison['traditional_ocr']['tesseract']['processing_time']:.2f}초")
        
        print("\n멀티모달 LLM:")
        print(f"  텍스트: {comparison['multimodal_llm']['text'][:100]}...")
        print(f"  처리시간: {comparison['multimodal_llm']['processing_time']:.2f}초")
        print(f"  비용: ${comparison['multimodal_llm']['cost']:.4f}")
        
    except Exception as e:
        print(f"비교 오류: {e}")
```

## Claude 3.5 Sonnet 기반 OCR 실습

### Anthropic Claude 클래스
```python
import anthropic
import base64

class ClaudeOCR:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def extract_text_claude(self, image_path):
        """Claude를 사용한 텍스트 추출"""
        
        # 이미지를 base64로 인코딩
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": """
이 이미지의 모든 텍스트를 정확히 추출하고 다음 JSON 형식으로 반환해주세요:

{
  "extracted_text": "전체 텍스트",
  "confidence": 0.95,
  "language": "korean|english|mixed",
  "document_type": "invoice|contract|receipt|form|letter",
  "key_information": {
    "dates": ["발견된 모든 날짜"],
    "amounts": ["발견된 모든 금액"],
    "organizations": ["회사명/기관명"],
    "contact_info": ["전화번호, 이메일 등"]
  }
}

모든 한글, 영문, 숫자를 놓치지 마세요.
"""
                        }
                    ]
                }
            ]
        )
        
        return message.content[0].text

# 사용 예제
if __name__ == "__main__":
    claude_ocr = ClaudeOCR()
    
    try:
        result = claude_ocr.extract_text_claude("sample_document.jpg")
        print("Claude 결과:")
        print(result)
    except Exception as e:
        print(f"Claude 오류: {e}")
```

## 통합 비교 및 평가 코드

### 종합 성능 비교 클래스
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class OCRComparison:
    def __init__(self):
        self.results = []
    
    def run_comprehensive_test(self, image_paths, ground_truths=None):
        """종합적인 OCR 성능 테스트"""
        
        # OCR 인스턴스 생성
        multimodal_ocr = MultimodalOCR()
        claude_ocr = ClaudeOCR()
        
        for i, image_path in enumerate(image_paths):
            print(f"테스트 중: {image_path}")
            
            # 각 OCR 엔진 테스트
            test_result = {
                'image': image_path,
                'ground_truth': ground_truths[i] if ground_truths else None
            }
            
            # GPT-4V 테스트
            try:
                gpt_result = multimodal_ocr.extract_text_with_coordinates(image_path)
                test_result['gpt4v'] = {
                    'text': gpt_result.get('extracted_text', ''),
                    'cost': gpt_result.get('processing_info', {}).get('estimated_cost', 0),
                    'tokens': gpt_result.get('processing_info', {}).get('tokens_used', 0)
                }
            except Exception as e:
                test_result['gpt4v'] = {'error': str(e)}
            
            # Claude 테스트 (실제 API 키가 있는 경우)
            try:
                claude_result = claude_ocr.extract_text_claude(image_path)
                test_result['claude'] = {
                    'text': claude_result,
                    'cost': 0.003,  # 추정값
                    'tokens': 1000   # 추정값
                }
            except Exception as e:
                test_result['claude'] = {'error': str(e)}
            
            # 전통적인 OCR과 비교
            try:
                comparison = multimodal_ocr.compare_with_traditional_ocr(image_path)
                test_result['paddleocr'] = comparison['traditional_ocr']['paddleocr']
                test_result['tesseract'] = comparison['traditional_ocr']['tesseract']
            except Exception as e:
                test_result['traditional_error'] = str(e)
            
            self.results.append(test_result)
        
        return self.results
    
    def generate_report(self, save_path='ocr_comparison_report.html'):
        """비교 결과 리포트 생성"""
        if not self.results:
            print("테스트 결과가 없습니다. run_comprehensive_test를 먼저 실행하세요.")
            return
        
        # 결과를 DataFrame으로 변환
        data = []
        for result in self.results:
            for engine in ['paddleocr', 'tesseract', 'gpt4v', 'claude']:
                if engine in result and 'error' not in result[engine]:
                    data.append({
                        'image': result['image'],
                        'engine': engine,
                        'text_length': len(result[engine].get('text', '')),
                        'processing_time': result[engine].get('processing_time', 0),
                        'cost': result[engine].get('cost', 0),
                        'tokens': result[engine].get('tokens', 0)
                    })
        
        df = pd.DataFrame(data)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 처리 시간 비교
        sns.boxplot(data=df, x='engine', y='processing_time', ax=axes[0,0])
        axes[0,0].set_title('처리 시간 비교')
        axes[0,0].set_ylabel('시간 (초)')
        
        # 비용 비교
        sns.boxplot(data=df, x='engine', y='cost', ax=axes[0,1])
        axes[0,1].set_title('비용 비교')
        axes[0,1].set_ylabel('비용 ($)')
        
        # 텍스트 길이 비교
        sns.boxplot(data=df, x='engine', y='text_length', ax=axes[1,0])
        axes[1,0].set_title('추출된 텍스트 길이')
        axes[1,0].set_ylabel('글자 수')
        
        # 토큰 사용량 (LLM만)
        llm_df = df[df['engine'].isin(['gpt4v', 'claude'])]
        if not llm_df.empty:
            sns.boxplot(data=llm_df, x='engine', y='tokens', ax=axes[1,1])
            axes[1,1].set_title('토큰 사용량 (LLM)')
            axes[1,1].set_ylabel('토큰 수')
        
        plt.tight_layout()
        plt.savefig('ocr_comparison_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # HTML 리포트 생성
        html_content = f"""
        <html>
        <head><title>OCR 성능 비교 리포트</title></head>
        <body>
        <h1>OCR 엔진 성능 비교</h1>
        
        <h2>요약 통계</h2>
        {df.groupby('engine').agg({
            'processing_time': 'mean',
            'cost': 'mean',
            'text_length': 'mean',
            'tokens': 'mean'
        }).round(4).to_html()}
        
        <h2>상세 결과</h2>
        {df.to_html(index=False)}
        
        <h2>결론</h2>
        <ul>
        <li>속도: {'가장 빠른 엔진 - ' + df.loc[df['processing_time'].idxmin(), 'engine'] if not df.empty else 'N/A'}</li>
        <li>비용: {'가장 경제적 - ' + df.loc[df['cost'].idxmin(), 'engine'] if not df.empty else 'N/A'}</li>
        <li>텍스트량: {'가장 많이 추출 - ' + df.loc[df['text_length'].idxmax(), 'engine'] if not df.empty else 'N/A'}</li>
        </ul>
        </body>
        </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"리포트가 {save_path}에 저장되었습니다.")

# 실행 예제
if __name__ == "__main__":
    comparison = OCRComparison()
    
    # 테스트 이미지들 (실제 사용시 경로 수정)
    test_images = [
        "test1_korean_invoice.jpg",
        "test2_english_contract.jpg", 
        "test3_mixed_receipt.jpg"
    ]
    
    # 정답 텍스트 (옵션)
    ground_truths = [
        "삼성전자 청구서 2025년 1월",
        "Service Agreement between A and B",
        "편의점 영수증 총 15,000원"
    ]
    
    # 종합 테스트 실행
    results = comparison.run_comprehensive_test(test_images, ground_truths)
    
    # 리포트 생성
    comparison.generate_report()
```

## 실습 진행 가이드

### 1. 준비 사항
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "ANTHROPIC_API_KEY=your_claude_key_here" >> .env
```

### 2. 실습 순서
1. **환경 설정**: 패키지 설치 및 API 키 설정
2. **기본 테스트**: 단일 이미지로 각 OCR 엔진 테스트
3. **성능 비교**: 여러 이미지로 종합 성능 측정
4. **결과 분석**: 정확도, 속도, 비용 종합 평가

### 3. 주의 사항
- API 사용량과 비용 모니터링
- 이미지 크기 최적화로 토큰 비용 절약
- 에러 처리와 재시도 로직 구현

---

**강의 섹션**: 7. GPT급 LLM을 통한 OCR 보정 (90분)
**슬라이드 번호**: 31-4/47
