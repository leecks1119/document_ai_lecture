# 슬라이드 28: 고급 전처리 기법들

**리딩 메시지**: "최신 AI 기반 전처리 기법으로 기존 방식 대비 30% 성능 향상을 달성하겠습니다"

## 딥러닝 기반 전처리

- **Super Resolution**: ESRGAN, Real-ESRGAN, SwinIR (2025년 최신)
  - 저해상도 이미지 → 고해상도 변환
  - 2배-8배 해상도 향상 가능
- **Document Dewarping**: DewarpNet, DocUNet, DocTr (2024년 신기술)
  - 구겨진 문서 → 평면 문서 변환
  - 사진으로 촬영한 책, 구겨진 영수증 처리
- **저조도 텍스트 강화**: Text in the Dark, LLFlow (2024년 연구)
  - 어두운 환경에서 촬영한 문서 선명화
  - 야간 촬영, 실내 조명 부족 상황 대응
- **Shadow Removal**: ShadowFormer, Mask R-CNN 기반
  - 그림자 영역 자동 검출 및 보정
  - 스마트폰 촬영 문서에서 특히 효과적

## 전통적 방법 vs AI 방법 성능 비교

- 기울기 보정: Hough Transform (2° 오차) vs CNN (0.5° 오차)
- 노이즈 제거: Gaussian Filter vs DnCNN (PSNR 3dB 향상)
- 해상도 개선: Bicubic (artifacts 발생) vs ESRGAN (자연스러운 결과)

---

**강의 섹션**: 6. 전처리를 통한 OCR 인식률 향상 체감 (90분)
**슬라이드 번호**: 28/44

