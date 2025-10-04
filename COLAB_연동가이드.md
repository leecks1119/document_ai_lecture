# GitHub-Colab 연동 가이드

이 가이드는 **수강생들이 강의를 들을 때** Google Colab에서 실습 노트북을 쉽게 사용할 수 있도록 설명합니다.

---

## 🎯 수강생용: 노트북 사용 방법

### 방법 1: Colab 배지 클릭 (가장 쉬움! ⭐)

1. GitHub 저장소 접속: https://github.com/leecks1119/document_ai_lecture
2. `notebooks/` 폴더로 이동
3. 원하는 노트북 클릭 (예: `Lab01_개발환경구축.ipynb`)
4. 노트북 상단의 **"Open in Colab"** 배지 클릭!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/main/notebooks/Lab01_개발환경구축.ipynb)

### 방법 2: 직접 URL 입력

Colab 주소창에 다음 형식으로 입력:
```
https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/main/notebooks/[파일명].ipynb
```

**예시:**
- Lab01: https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/main/notebooks/Lab01_개발환경구축.ipynb
- Lab04: https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/main/notebooks/Lab04_OCR엔진비교.ipynb

### 방법 3: Colab에서 GitHub 저장소 열기

1. [Google Colab](https://colab.research.google.com/) 접속
2. `파일` → `노트북 열기`
3. `GitHub` 탭 클릭
4. 저장소 URL 입력: `leecks1119/document_ai_lecture`
5. 원하는 노트북 선택

---

## 📝 실습 진행 방법

### 1단계: GPU 설정 (권장)

더 빠른 실습을 위해 GPU를 활성화하세요:

```
런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU 선택 → 저장
```

### 2단계: 셀 순서대로 실행

**중요!** 노트북은 위에서 아래로 순서대로 실행해야 합니다.

**실행 방법:**
- `Shift + Enter`: 현재 셀 실행 후 다음 셀로 이동
- `Ctrl + Enter`: 현재 셀만 실행
- 또는 각 셀 왼쪽의 ▶️ 버튼 클릭

### 3단계: 결과 확인

- 코드 셀 아래에 출력 결과가 표시됩니다
- 그래프와 이미지는 자동으로 표시됩니다

---

## 💡 자주 묻는 질문 (FAQ)

### Q1: Google Drive 마운트를 꼭 해야 하나요?

**A: 아니요, 선택사항입니다!**

**Drive 마운트가 필요한 경우:**
- ✅ 여러 날에 걸쳐 작업할 때
- ✅ 대용량 파일을 저장하고 싶을 때
- ✅ 결과를 계속 보관하고 싶을 때

**Drive 마운트가 불필요한 경우:**
- ✅ 한 번에 끝나는 짧은 실습
- ✅ 결과를 다운로드만 받으면 되는 경우

**Drive 마운트 방법:**
```python
from google.colab import drive
drive.mount('/content/drive')

# 작업 폴더 생성
import os
SAVE_DIR = '/content/drive/MyDrive/DocumentAI_Results'
os.makedirs(SAVE_DIR, exist_ok=True)
os.chdir(SAVE_DIR)
```

### Q2: 각자의 Drive가 연결되나요?

**A: 네! 본인의 Google Drive만 연결됩니다.**

- 김철수 학생 → 김철수의 Drive
- 이영희 학생 → 이영희의 Drive
- 서로의 Drive는 보이지 않습니다 (개인 저장공간)

### Q3: 세션이 자동으로 종료되나요?

**A: 네, Colab 무료 버전은 제한이 있습니다.**

- **세션 시간 제한**: 12시간 (Pro: 24시간)
- **유휴 타임아웃**: 90분 무활동 시 종료
- **파일 휘발성**: `/content` 폴더는 세션 종료 시 삭제

**해결책:**
- 중요한 결과는 Drive에 저장
- 또는 로컬로 다운로드

### Q4: 패키지 설치가 실패했어요

**해결 방법:**
```python
# 캐시 삭제 후 재설치
!pip cache purge
!pip install --no-cache-dir git+https://github.com/leecks1119/document_ai_lecture.git
```

### Q5: GPU 메모리가 부족해요

**해결 방법:**
```
런타임 → 런타임 다시 시작
```

---

## 🎨 Colab 기본 사용법

### 셀 타입

**코드 셀 (Code Cell)**
```python
# Python 코드를 실행합니다
print("Hello, Document AI!")
```

**텍스트 셀 (Markdown Cell)**
```markdown
# 제목
설명을 작성합니다
```

### 단축키

| 단축키 | 기능 |
|--------|------|
| `Shift + Enter` | 셀 실행 후 다음 셀로 |
| `Ctrl + Enter` | 현재 셀만 실행 |
| `Ctrl + M + B` | 아래에 코드 셀 삽입 |
| `Ctrl + M + M` | 마크다운 셀로 변경 |
| `Ctrl + M + Y` | 코드 셀로 변경 |
| `Ctrl + M + D` | 셀 삭제 |

### Shell 명령어

Colab에서는 `!`를 붙여 Shell 명령어를 실행할 수 있습니다:

```python
!pwd                    # 현재 디렉토리 확인
!ls                     # 파일 목록 확인
!pip install package    # 패키지 설치
!wget URL               # 파일 다운로드
```

---

## 🔧 문제 해결

### 한글 폰트 깨짐

```python
!apt-get install -y fonts-nanum
!fc-cache -fv

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
```

### Tesseract 한글 인식 안 됨

```python
!apt-get install -y tesseract-ocr-kor
```

### 파일 업로드 방법

```python
from google.colab import files

# 파일 업로드
uploaded = files.upload()

# 업로드된 파일명 확인
for filename in uploaded.keys():
    print(f'업로드된 파일: {filename}')
```

### 파일 다운로드 방법

```python
from google.colab import files

# 파일 다운로드
files.download('result.csv')
```

---

## 📊 실습 완료 체크리스트

각 실습 완료 후 체크하세요:

- [ ] Lab01: 개발환경 구축
- [ ] Lab02: Document AI 기술표
- [ ] Lab03: PaddleOCR 기본 사용
- [ ] Lab04: OCR 엔진 비교
- [ ] Lab05: 신뢰도 측정
- [ ] Lab06: 이미지 전처리
- [ ] Lab07: OCR 앙상블
- [ ] Lab08: 표 검출
- [ ] Lab09: NER 정보 추출
- [ ] Lab10: Cursor AI 프로젝트
- [ ] Lab11: 전체 시스템 테스트

---

## 🎓 강사용: 노트북 관리 방법

### 새 노트북 추가 방법

1. 로컬에서 `.ipynb` 파일 생성
2. GitHub에 push
3. 노트북 첫 셀에 배지 추가:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leecks1119/document_ai_lecture/blob/main/notebooks/[파일명].ipynb)
```

### 노트북 업데이트 방법

1. 로컬에서 노트북 수정
2. GitHub에 push
3. 수강생들은 자동으로 최신 버전 사용

---

## 📞 지원

- **GitHub Issues**: https://github.com/leecks1119/document_ai_lecture/issues
- **문제 보고 시 포함사항**:
  - 실습 번호
  - 에러 메시지
  - 실행 환경 (GPU 여부 등)

---

**즐거운 실습 되세요! 🚀**

