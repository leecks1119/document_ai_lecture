# 슬라이드 31-3: Projection Layer와 대조 학습

**리딩 메시지**: "이미지와 텍스트를 같은 좌표계로 맞추는 핵심 기술을 이해하면 멀티모달 AI의 작동 원리가 보입니다"

## Projection Layer (투사 층)

### 목적과 역할
- **문제**: 이미지 임베딩과 텍스트 임베딩이 서로 다른 좌표계에 존재
- **해결**: 이미지 임베딩을 텍스트 임베딩 공간으로 "번역"
- **비유**: 다른 나라 언어를 같은 언어로 번역하는 역할

### 수학적 변환
```python
# 선형 변환 예시
x_proj = W · x_img + b

# W: [768 x 1024] 가중치 행렬
# b: [768] 편향 벡터
# x_img: 1024차원 이미지 임베딩
# x_proj: 768차원으로 변환된 벡터
```

### MLP 방식 (더 복잡한 변환)
```python
# 2층 MLP 변환
x_proj = σ(W2 · σ(W1 · x_img + b1) + b2)
# σ: 비선형 활성화 함수 (ReLU, GELU 등)
```

## 대조 학습 (Contrastive Learning)

### 기본 개념
- **목적**: 같은 의미의 이미지-텍스트는 가깝게, 다른 의미는 멀게
- **비유**: 가족 사진에서 이름 맞추기 게임
- **방법**: Positive Pair는 가깝게, Negative Pair는 멀리

### 학습 데이터 구성
```
Positive Pair (같은 의미):
- "A" 글자 이미지 ↔ "A" 텍스트 토큰
- "계약서" 이미지 ↔ "계약서" 텍스트

Negative Pair (다른 의미):
- "A" 글자 이미지 ↔ "B" 텍스트 토큰
- "계약서" 이미지 ↔ "영수증" 텍스트
```

### InfoNCE 손실 함수
```
L = -log(exp(sim(vi, ti)/τ) / Σj exp(sim(vi, tj)/τ))

여기서:
- vi: i번째 이미지 임베딩
- ti: i번째 텍스트 임베딩  
- sim: 코사인 유사도
- τ: 온도 파라미터 (분포 날카로움 조절)
```

## 전체 학습 과정

### 1단계: 임베딩 생성
```
이미지 → CNN/ViT → x_img (1024차원)
텍스트 → LLM → x_txt (768차원)
```

### 2단계: 차원 맞추기
```
x_img → Projection Layer → x_proj (768차원)
```

### 3단계: 대조 학습으로 정렬
```
같은 의미: x_proj와 x_txt 거리 최소화
다른 의미: x_proj와 x_txt 거리 최대화
```

### 4단계: 결과
```
"A" 이미지 벡터 ≈ "A" 텍스트 벡터 (같은 공간에서 가까움)
"B" 이미지 벡터 ≈ "B" 텍스트 벡터 (같은 공간에서 가까움)
```

## OCR 적용 시 활용

### 문자 인식 과정
1. **이미지 입력**: "A" 글자가 포함된 문서 이미지
2. **비전 인코딩**: 이미지를 패치 임베딩으로 변환
3. **Projection**: 이미지 임베딩을 텍스트 공간으로 투사
4. **매칭**: 가장 가까운 텍스트 토큰 찾기 → "A" 출력

### 코드 예시
```python
def recognize_character(image_patch):
    # 1. 이미지 임베딩 생성
    img_embedding = vision_encoder(image_patch)
    
    # 2. Projection Layer 적용
    projected_embedding = projection_layer(img_embedding)
    
    # 3. 모든 문자 토큰과 유사도 계산
    similarities = []
    for char in ["A", "B", "C", ...]:
        char_embedding = text_encoder(char)
        similarity = cosine_similarity(projected_embedding, char_embedding)
        similarities.append((char, similarity))
    
    # 4. 가장 유사한 문자 반환
    return max(similarities, key=lambda x: x[1])[0]
```

## 실무 최적화 팁

### 비용 절감
- **토큰 수 최소화**: 불필요한 설명 제거
- **배치 처리**: 여러 문서를 한 번에 처리
- **캐싱**: 유사한 패턴 결과 재사용

### 성능 향상
- **해상도 조절**: 중요 영역만 고해상도 처리
- **다중 스케일**: 여러 해상도에서 처리 후 융합
- **적응적 임계값**: 신뢰도 기반 후처리 적용

## 한계와 고려사항

### 환각(Hallucination) 문제
- **원인**: 문맥 보정 과정에서 없는 글자 생성
- **대응**: 신뢰도 임계값, 검증 규칙 적용

### 좌표 정보 부재
- **문제**: 기본적으로 위치 정보 미제공
- **해결**: Grounding 헤드 추가 또는 하이브리드 접근

---

**강의 섹션**: 7. GPT급 LLM을 통한 OCR 보정 (90분)
**슬라이드 번호**: 31-3/47
