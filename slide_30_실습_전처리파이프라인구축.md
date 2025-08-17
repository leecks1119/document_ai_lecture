# 슬라이드 30: 실습 - 전처리 파이프라인 구축

**리딩 메시지**: "실제 업무 문서로 완전한 전처리 파이프라인을 구축하고 성능을 측정해보겠습니다"

## 실습 시나리오
회사 계약서 디지털화 프로젝트
- **문제 상황**: 스캔 품질 불량, 기울어짐, 그림자, 저해상도
- **목표**: OCR 정확도 90% 이상 달성

## 실습 단계

1. **현황 분석**: 
   - 100개 샘플 문서 품질 평가
   - 베이스라인 OCR 정확도 측정 (예상: 70-75%)

2. **전처리 파이프라인 구현**:

### 완전한 전처리 파이프라인 코드
```python
import cv2
import numpy as np
import math
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

class DocumentPreprocessor:
    def __init__(self):
        self.debug_mode = True
        self.processed_images = {}
    
    def assess_image_quality(self, image):
        """이미지 품질 자동 평가"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. 해상도 분석
        height, width = gray.shape
        dpi_estimate = width * height / (8.5 * 11 * 300)  # A4 기준 DPI 추정
        
        # 2. 선명도 측정 (Laplacian Variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 3. 조명 균일성 (히스토그램 분석)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        brightness_variance = np.var(hist)
        
        # 4. 기울기 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        skew_angle = 0
        if lines is not None:
            angles = []
            for rho, theta in lines[:20]:  # 상위 20개 라인만 분석
                angle = theta * 180 / np.pi
                if 45 < angle < 135:  # 수직선 제외
                    angles.append(angle - 90)
            if angles:
                skew_angle = np.median(angles)
        
        # 5. 노이즈 레벨 측정
        noise_level = np.std(cv2.medianBlur(gray, 5) - gray) / 255.0
        
        quality_metrics = {
            'dpi_estimate': dpi_estimate,
            'blur_score': blur_score,
            'brightness_variance': brightness_variance,
            'skew_angle': abs(skew_angle),
            'noise_level': noise_level,
            'width': width,
            'height': height
        }
        
        if self.debug_mode:
            print(f"이미지 품질 평가:")
            print(f"  추정 DPI: {dpi_estimate:.1f}")
            print(f"  선명도 점수: {blur_score:.1f} {'(흐림)' if blur_score < 100 else '(선명)'}")
            print(f"  기울기: {abs(skew_angle):.2f}도")
            print(f"  노이즈 레벨: {noise_level:.3f}")
        
        return quality_metrics
    
    def correct_skew(self, image, angle):
        """기울기 보정"""
        if abs(angle) < 0.5:  # 0.5도 미만은 보정하지 않음
            return image
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 회전 변환 행렬 생성
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 회전 후 이미지 크기 계산
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 변환 행렬 조정
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 이미지 회전
        corrected = cv2.warpAffine(image, M, (new_w, new_h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        
        return corrected
    
    def enhance_sharpness(self, image):
        """이미지 선명도 향상"""
        if len(image.shape) == 3:
            # 컬러 이미지
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Sharpness(pil_image)
            enhanced = enhancer.enhance(2.0)  # 선명도 2배 증가
            return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        else:
            # 그레이스케일 이미지
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            return cv2.filter2D(image, -1, kernel)
    
    def denoise_image(self, image):
        """노이즈 제거"""
        if len(image.shape) == 3:
            # 컬러 이미지
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # 그레이스케일 이미지
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def enhance_contrast(self, image):
        """대비 향상"""
        if len(image.shape) == 3:
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE 적용 (L 채널만)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # 채널 합치기
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # 그레이스케일 이미지
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image)
    
    def binarize_adaptive(self, image):
        """적응적 이진화"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 여러 이진화 방법 시도
        methods = {
            'adaptive_gaussian': cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            ),
            'adaptive_mean': cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            ),
            'otsu': cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        }
        
        # 가장 좋은 결과 선택 (텍스트 영역이 많은 것)
        best_method = 'adaptive_gaussian'
        max_text_area = 0
        
        for method_name, binary_img in methods.items():
            # 연결된 구성요소 분석
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                255 - binary_img, connectivity=8
            )
            
            # 텍스트로 추정되는 영역 계산
            text_area = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # 텍스트 특성 (적당한 크기, 가로세로 비율)
                if 20 < area < 5000 and 0.1 < height/width < 10:
                    text_area += area
            
            if text_area > max_text_area:
                max_text_area = text_area
                best_method = method_name
        
        return methods[best_method]
    
    def remove_shadows(self, image):
        """그림자 제거"""
        if len(image.shape) == 3:
            # RGB로 변환
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 그림자 마스크 생성
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            
            # 그림자 보정
            diff_img = 255 - cv2.absdiff(gray, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            
            return cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
        else:
            return image
    
    def super_resolution(self, image, scale_factor=2):
        """슈퍼 해상도 (간단한 버전)"""
        height, width = image.shape[:2]
        new_height, new_width = height * scale_factor, width * scale_factor
        
        # Bicubic 업샘플링
        upsampled = cv2.resize(image, (new_width, new_height), 
                              interpolation=cv2.INTER_CUBIC)
        
        # 언샤프 마스킹으로 디테일 강화
        gaussian = cv2.GaussianBlur(upsampled, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(upsampled, 1.5, gaussian, -0.5, 0)
        
        return unsharp_mask
    
    def adaptive_preprocessing_pipeline(self, image_path):
        """적응적 전처리 파이프라인"""
        # 원본 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        original = image.copy()
        self.processed_images['01_original'] = original
        
        # 1. 품질 평가
        quality_metrics = self.assess_image_quality(image)
        
        # 2. 그림자 제거 (필요시)
        if quality_metrics['brightness_variance'] > 1000:
            image = self.remove_shadows(image)
            self.processed_images['02_shadow_removed'] = image
            print("✓ 그림자 제거 적용")
        
        # 3. 슈퍼 해상도 (저해상도인 경우)
        if quality_metrics['dpi_estimate'] < 200:
            image = self.super_resolution(image, scale_factor=2)
            self.processed_images['03_super_resolution'] = image
            print("✓ 슈퍼 해상도 적용")
        
        # 4. 기울기 보정 (필요시)
        if quality_metrics['skew_angle'] > 1.0:
            image = self.correct_skew(image, quality_metrics['skew_angle'])
            self.processed_images['04_skew_corrected'] = image
            print(f"✓ 기울기 보정 적용: {quality_metrics['skew_angle']:.2f}도")
        
        # 5. 노이즈 제거 (필요시)
        if quality_metrics['noise_level'] > 0.02:
            image = self.denoise_image(image)
            self.processed_images['05_denoised'] = image
            print("✓ 노이즈 제거 적용")
        
        # 6. 선명도 향상 (필요시)
        if quality_metrics['blur_score'] < 100:
            image = self.enhance_sharpness(image)
            self.processed_images['06_sharpened'] = image
            print("✓ 선명도 향상 적용")
        
        # 7. 대비 향상
        image = self.enhance_contrast(image)
        self.processed_images['07_contrast_enhanced'] = image
        print("✓ 대비 향상 적용")
        
        # 8. 이진화 (OCR 최적화)
        binary = self.binarize_adaptive(image)
        self.processed_images['08_binarized'] = binary
        print("✓ 적응적 이진화 적용")
        
        return image, binary, quality_metrics
    
    def visualize_results(self, save_path='preprocessing_results.png'):
        """전처리 결과 시각화"""
        num_images = len(self.processed_images)
        cols = 3
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (stage, img) in enumerate(self.processed_images.items()):
            row, col = idx // cols, idx % cols
            
            if len(img.shape) == 3:
                # BGR to RGB 변환
                display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                display_img = img
                
            axes[row, col].imshow(display_img, cmap='gray' if len(img.shape) == 2 else None)
            axes[row, col].set_title(stage.replace('_', ' ').title())
            axes[row, col].axis('off')
        
        # 빈 subplot 숨기기
        for idx in range(num_images, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"결과 이미지가 {save_path}에 저장되었습니다.")

# 사용 예제
if __name__ == "__main__":
    preprocessor = DocumentPreprocessor()
    
    # 이미지 경로 (실제 사용 시 수정)
    image_path = "sample_document.jpg"
    
    try:
        # 전처리 파이프라인 실행
        processed_img, binary_img, metrics = preprocessor.adaptive_preprocessing_pipeline(image_path)
        
        # 결과 시각화
        preprocessor.visualize_results()
        
        # 처리된 이미지 저장
        cv2.imwrite('processed_document.jpg', processed_img)
        cv2.imwrite('binary_document.jpg', binary_img)
        
        print("\n전처리 완료!")
        print("처리된 이미지: processed_document.jpg")
        print("이진화 이미지: binary_document.jpg")
        
    except Exception as e:
        print(f"오류 발생: {e}")
```

### OCR 성능 비교 코드
```python
# 전처리 전후 OCR 성능 비교
import pytesseract
from paddleocr import PaddleOCR

def compare_ocr_performance(original_path, processed_img, binary_img):
    """전처리 전후 OCR 성능 비교"""
    
    # OCR 엔진 초기화
    paddleocr = PaddleOCR(use_angle_cls=True, lang='korean')
    
    results = {}
    
    # 1. 원본 이미지 OCR
    print("원본 이미지 OCR 실행...")
    original = cv2.imread(original_path)
    
    # Tesseract
    original_text_tess = pytesseract.image_to_string(original, lang='kor+eng')
    
    # PaddleOCR
    paddle_results = paddleocr.ocr(original_path, cls=True)
    original_text_paddle = ' '.join([
        word_info[1][0] for line in paddle_results for word_info in line
    ])
    
    results['original'] = {
        'tesseract': original_text_tess.strip(),
        'paddleocr': original_text_paddle.strip()
    }
    
    # 2. 전처리된 이미지 OCR
    print("전처리된 이미지 OCR 실행...")
    
    # Tesseract
    processed_text_tess = pytesseract.image_to_string(processed_img, lang='kor+eng')
    
    # PaddleOCR - 임시 파일로 저장 후 처리
    cv2.imwrite('temp_processed.jpg', processed_img)
    paddle_results = paddleocr.ocr('temp_processed.jpg', cls=True)
    processed_text_paddle = ' '.join([
        word_info[1][0] for line in paddle_results for word_info in line
    ])
    
    results['processed'] = {
        'tesseract': processed_text_tess.strip(),
        'paddleocr': processed_text_paddle.strip()
    }
    
    # 3. 이진화된 이미지 OCR
    print("이진화된 이미지 OCR 실행...")
    
    # Tesseract
    binary_text_tess = pytesseract.image_to_string(binary_img, lang='kor+eng')
    
    # PaddleOCR
    cv2.imwrite('temp_binary.jpg', binary_img)
    paddle_results = paddleocr.ocr('temp_binary.jpg', cls=True)
    binary_text_paddle = ' '.join([
        word_info[1][0] for line in paddle_results for word_info in line
    ])
    
    results['binary'] = {
        'tesseract': binary_text_tess.strip(),
        'paddleocr': binary_text_paddle.strip()
    }
    
    # 결과 출력
    print("\n" + "="*60)
    print("OCR 성능 비교 결과")
    print("="*60)
    
    for stage, engines in results.items():
        print(f"\n[{stage.upper()}]")
        for engine, text in engines.items():
            print(f"{engine}: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    return results

# 실행 예제
if __name__ == "__main__":
    # 전처리 실행
    preprocessor = DocumentPreprocessor()
    image_path = "sample_document.jpg"
    
    processed_img, binary_img, metrics = preprocessor.adaptive_preprocessing_pipeline(image_path)
    
    # OCR 성능 비교
    ocr_results = compare_ocr_performance(image_path, processed_img, binary_img)
```

3. **성능 측정 및 비교**:
   - Before: 원본 이미지 → OCR
   - After: 전처리 → OCR
   - 정확도, 처리시간, 비용 종합 분석

4. **최적화 및 튜닝**:
   - 파라미터 조정으로 추가 성능 향상
   - 처리 시간 vs 정확도 트레이드오프 분석

## 예상 성과

- OCR 정확도: 75% → 92% (17%p 향상)
- 재작업 건수: 25% → 8% (68% 감소)
- 투자 회수 기간: 3개월

---

**강의 섹션**: 6. 전처리를 통한 OCR 인식률 향상 체감 (90분)
**슬라이드 번호**: 30/44

