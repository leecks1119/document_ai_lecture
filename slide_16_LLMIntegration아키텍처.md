# 슬라이드 16: LLM Integration 아키텍처

**리딩 메시지**: "LLM과 OCR의 효과적 결합이 차세대 Document AI의 핵심입니다"

## 통합 방식

- **Sequential Processing**: OCR → LLM 순차 처리
- **Parallel Processing**: OCR + LLM 동시 처리 후 결과 융합
- **Feedback Loop**: LLM 결과를 OCR 재처리에 활용
- **비용 최적화**:
  - GPT-4 API: $0.03/1K tokens
  - 로컬 LLaMA: 초기 구축비용 高, 운영비용 低
  - 하이브리드 접근: 간단한 보정은 로컬, 복잡한 추론은 API

---

**강의 섹션**: 3. 기술요소 원리 (90분)
**슬라이드 번호**: 16/44

