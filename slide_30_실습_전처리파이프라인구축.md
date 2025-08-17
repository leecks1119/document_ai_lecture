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
   ```python
   def contract_preprocessing_pipeline(image_path):
       # 1. 이미지 로드 및 기본 검증
       image = cv2.imread(image_path)
       quality_metrics = assess_quality(image)
       
       # 2. 적응적 전처리 적용
       if quality_metrics['skew_angle'] > 1.0:
           image = correct_skew(image, quality_metrics['skew_angle'])
       
       if quality_metrics['blur_score'] < 100:
           image = sharpen_image(image)
       
       if quality_metrics['noise_level'] > 0.3:
           image = denoise(image)
       
       # 3. OCR 최적화
       image = enhance_contrast(image)
       image = binarize_adaptive(image)
       
       return image
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

