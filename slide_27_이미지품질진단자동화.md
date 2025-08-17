# 슬라이드 27: 이미지 품질 진단 자동화

**리딩 메시지**: "이미지 품질을 사전 진단하여 적응적 전처리 파이프라인을 구축하겠습니다"

## 품질 지표 자동 측정

- **해상도 분석**: DPI 계산, 픽셀 밀도 체크
- **선명도 측정**: Laplacian Variance, Sobel Edge Detection
- **조명 균일성**: 히스토그램 분석, 대비 측정
- **기울기 검출**: Hough Line Transform
- **노이즈 레벨**: Signal-to-Noise Ratio 계산

## 적응적 처리 전략

```python
def adaptive_preprocessing(image):
    quality_score = assess_image_quality(image)
    
    if quality_score['blur'] > 0.3:
        image = apply_sharpening(image)
    if quality_score['skew'] > 2.0:
        image = correct_skew(image)
    if quality_score['noise'] > 0.2:
        image = denoise(image)
    
    return image
```

---

**강의 섹션**: 6. 전처리를 통한 OCR 인식률 향상 체감 (90분)
**슬라이드 번호**: 27/44

