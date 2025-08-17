# 슬라이드 23: OCR 정확도 측정 및 평가 방법론

**리딩 메시지**: "과학적인 평가 체계로 OCR 시스템의 지속적 개선이 가능합니다"

## 평가 지표

- **Character Accuracy**: (총 문자 수 - 에러 문자 수) / 총 문자 수
- **Word Accuracy**: (총 단어 수 - 에러 단어 수) / 총 단어 수  
- **Edit Distance**: Levenshtein Distance 기반 유사도
- **Confidence Score**: OCR 엔진의 확신도 지표

## 평가 프로세스

1. **Ground Truth 구축**: 수동 라벨링 (외주 vs 내부)
2. **테스트셋 구성**: 업무 문서 유형별 대표 샘플
3. **통계적 검증**: 신뢰구간, A/B 테스트
4. **지속적 모니터링**: 운영 환경에서의 성능 추적

---

**강의 섹션**: 5. OCR 기본 (75분)
**슬라이드 번호**: 23/47

## 완전한 성능 평가 및 벤치마크 코드

### 환경 설정
```bash
# 필요한 패키지 설치
pip install pandas numpy matplotlib seaborn
pip install sklearn editdistance python-Levenshtein
pip install plotly dash  # 인터랙티브 시각화
pip install jiwer  # Word Error Rate 계산
```

### 종합 평가 시스템 클래스
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import editdistance
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from collections import defaultdict, Counter
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

# Word Error Rate 계산용
try:
    import jiwer
except ImportError:
    print("jiwer 패키지가 없습니다. pip install jiwer 로 설치하세요.")

@dataclass
class EvaluationMetrics:
    """평가 지표 클래스"""
    character_accuracy: float
    word_accuracy: float
    edit_distance: int
    normalized_edit_distance: float
    word_error_rate: float
    confidence_score: float
    processing_time: float
    cost: float
    
class OCRBenchmarkSuite:
    """OCR 성능 평가 및 벤치마크 시스템"""
    
    def __init__(self):
        self.results = []
        self.error_patterns = defaultdict(int)
        self.confusion_matrix = defaultdict(int)
        
    def calculate_character_accuracy(self, predicted: str, ground_truth: str) -> Tuple[float, int]:
        """문자 단위 정확도 계산"""
        if not ground_truth:
            return 0.0, 0
        
        # Edit distance 계산
        edit_dist = editdistance.eval(predicted, ground_truth)
        
        # 정확도 계산 (1 - (edit distance / max length))
        max_length = max(len(predicted), len(ground_truth))
        accuracy = 1 - (edit_dist / max_length) if max_length > 0 else 0
        
        return max(0, accuracy * 100), edit_dist
    
    def calculate_word_accuracy(self, predicted: str, ground_truth: str) -> float:
        """단어 단위 정확도 계산"""
        pred_words = predicted.split()
        true_words = ground_truth.split()
        
        if not true_words:
            return 0.0
        
        # 단어별 매칭
        correct_words = 0
        for i, true_word in enumerate(true_words):
            if i < len(pred_words) and pred_words[i] == true_word:
                correct_words += 1
        
        return (correct_words / len(true_words)) * 100
    
    def calculate_word_error_rate(self, predicted: str, ground_truth: str) -> float:
        """Word Error Rate (WER) 계산"""
        try:
            # jiwer 사용 (설치되어 있는 경우)
            return jiwer.wer(ground_truth, predicted) * 100
        except:
            # 수동 계산
            pred_words = predicted.split()
            true_words = ground_truth.split()
            
            if not true_words:
                return 0.0
            
            # Levenshtein distance for words
            edit_dist = editdistance.eval(pred_words, true_words)
            return (edit_dist / len(true_words)) * 100
    
    def run_comprehensive_evaluation(self, test_dataset: List[Dict]) -> Dict:
        """종합 평가 실행
        
        Args:
            test_dataset: [{'image_path': str, 'ground_truth': str, 'document_type': str}, ...]
        """
        
        # OCR 엔진들 (이전 슬라이드의 클래스들 import 필요)
        from ocr_benchmark import OCRBenchmark  # 이전 코드
        from hybrid_system import HybridDocumentAI  # 이전 코드
        
        benchmark = OCRBenchmark()
        hybrid_ai = HybridDocumentAI()
        
        engines_results = defaultdict(list)
        
        print(f"총 {len(test_dataset)}개 문서 평가 시작...")
        
        for i, data in enumerate(test_dataset):
            image_path = data['image_path']
            ground_truth = data['ground_truth']
            doc_type = data.get('document_type', 'unknown')
            
            print(f"진행률: {i+1}/{len(test_dataset)} - {image_path}")
            
            try:
                # 전통 OCR 엔진들 테스트
                ocr_results = benchmark.run_comparison(image_path, ground_truth)
                
                for engine_result in ocr_results:
                    if 'error' not in engine_result:
                        char_acc, edit_dist = self.calculate_character_accuracy(
                            engine_result['text'], ground_truth)
                        word_acc = self.calculate_word_accuracy(
                            engine_result['text'], ground_truth)
                        wer = self.calculate_word_error_rate(
                            engine_result['text'], ground_truth)
                        
                        engines_results[engine_result['engine']].append({
                            'character_accuracy': char_acc,
                            'word_accuracy': word_acc,
                            'word_error_rate': wer,
                            'processing_time': engine_result.get('processing_time', 0),
                            'cost': engine_result.get('cost', 0)
                        })
                
            except Exception as e:
                print(f"평가 오류 - {image_path}: {e}")
                continue
        
        # 결과 집계
        summary_results = {}
        for engine, metrics_list in engines_results.items():
            if metrics_list:
                summary_results[engine] = {
                    'count': len(metrics_list),
                    'avg_character_accuracy': np.mean([m['character_accuracy'] for m in metrics_list]),
                    'avg_word_accuracy': np.mean([m['word_accuracy'] for m in metrics_list]),
                    'avg_word_error_rate': np.mean([m['word_error_rate'] for m in metrics_list]),
                    'avg_processing_time': np.mean([m['processing_time'] for m in metrics_list]),
                    'total_cost': sum([m['cost'] for m in metrics_list]),
                    'std_character_accuracy': np.std([m['character_accuracy'] for m in metrics_list])
                }
        
        return summary_results
    
    def generate_performance_report(self, results: Dict, save_path: str = "performance_report.html"):
        """성능 리포트 생성"""
        
        # 결과를 DataFrame으로 변환
        df_data = []
        for engine, stats in results.items():
            df_data.append({
                'Engine': engine,
                'Count': stats['count'],
                'Char_Accuracy': f"{stats['avg_character_accuracy']:.2f}%",
                'Word_Accuracy': f"{stats['avg_word_accuracy']:.2f}%",
                'Word_Error_Rate': f"{stats['avg_word_error_rate']:.2f}%",
                'Avg_Time': f"{stats['avg_processing_time']:.2f}s",
                'Total_Cost': f"${stats['total_cost']:.4f}",
                'Accuracy_StdDev': f"{stats['std_character_accuracy']:.2f}"
            })
        
        df = pd.DataFrame(df_data)
        
        # HTML 리포트 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OCR 성능 평가 리포트</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .highlight {{ background-color: #e8f5e8; }}
                .summary {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>OCR 엔진 성능 평가 리포트</h1>
            <div class="summary">
                <h3>평가 개요</h3>
                <p>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>총 평가 엔진 수: {len(results)}</p>
                <p>최고 성능 엔진 (문자 정확도): {max(results.items(), key=lambda x: x[1]['avg_character_accuracy'])[0]}</p>
                <p>최고 속도 엔진: {min(results.items(), key=lambda x: x[1]['avg_processing_time'])[0]}</p>
            </div>
            
            <h2>상세 성능 비교</h2>
            {df.to_html(classes='performance-table', index=False)}
            
            <h2>권장사항</h2>
            <ul>
                <li><strong>높은 정확도 우선</strong>: {max(results.items(), key=lambda x: x[1]['avg_character_accuracy'])[0]} 엔진 사용</li>
                <li><strong>빠른 처리 우선</strong>: {min(results.items(), key=lambda x: x[1]['avg_processing_time'])[0]} 엔진 사용</li>
                <li><strong>비용 효율성</strong>: 무료 엔진 중에서는 정확도가 높은 엔진 선택</li>
                <li><strong>하이브리드 접근</strong>: 품질에 따라 엔진을 적응적으로 선택하는 시스템 구축 권장</li>
            </ul>
        </body>
        </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"성능 리포트가 {save_path}에 저장되었습니다.")

# 실행 예제
if __name__ == "__main__":
    # 평가 시스템 초기화
    evaluator = OCRBenchmarkSuite()
    
    # 테스트 데이터셋 (실제 사용시 경로와 정답 수정)
    test_dataset = [
        {
            'image_path': 'test_images/invoice1.jpg',
            'ground_truth': '삼성전자 주식회사 청구서 2025년 1월 15일 총액 1,500,000원',
            'document_type': 'invoice'
        },
        {
            'image_path': 'test_images/contract1.jpg', 
            'ground_truth': '계약서 갑: 주식회사 ABC 을: 주식회사 XYZ 계약일: 2025-01-15',
            'document_type': 'contract'
        }
        # 실제로는 100개 이상의 다양한 문서 사용
    ]
    
    # 종합 평가 실행
    print("=== OCR 엔진 종합 평가 시작 ===")
    summary_results = evaluator.run_comprehensive_evaluation(test_dataset)
    
    # 결과 출력 및 리포트 생성
    evaluator.generate_performance_report(summary_results)
    
    print("평가 완료! 리포트 파일을 확인하세요.")
```

