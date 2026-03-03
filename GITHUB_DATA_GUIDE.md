# GitHub ChatBot 데이터 공유 가이드

## 개요
이 프로젝트에서 사용하는 챗봇 데이터를 GitHub에서 공유하는 방법을 설명합니다.

## 1. 데이터 준비 (원본 저장소)

### 1.1 온라인 저장소 사용 (권장)

**songys/Chatbot_data 사용 (공개 데이터)**

```bash
cd /workspace/chatbot_project/data/raw
wget https://github.com/songys/Chatbot_data/raw/master/ChatbotData.csv
```

또는 git clone:

```bash
cd /workspace/chatbot_project/data
git clone https://github.com/songys/Chatbot_data.git
cp Chatbot_data/ChatbotData.csv raw/
```

### 1.2 개인 저장소에서 데이터 가져오기

**당신의 GitHub 저장소에서:**

```bash
# SSH 키 기반 (권장)
cd /workspace/chatbot_project/data/raw
git clone git@github.com:YOUR_USERNAME/chatbot_data.git
cp chatbot_data/ChatbotData.csv .

# 또는 HTTPS 기반
git clone https://github.com/YOUR_USERNAME/chatbot_data.git
cp chatbot_data/ChatbotData.csv .

# 또는 직접 파일 다운로드
wget https://raw.githubusercontent.com/YOUR_USERNAME/chatbot_data/main/ChatbotData.csv
```

## 2. 프로젝트에 데이터 통합

### 2.1 Git 서브모듈 방식 (권장)

프로젝트 의존성을 명확하게 관리할 수 있습니다.

```bash
cd /workspace/chatbot_project

# 마지막 줄에 추가
git submodule add https://github.com/YOUR_USERNAME/chatbot_data.git data/chatbot_data

# 또는 공개 데이터 사용
git submodule add https://github.com/songys/Chatbot_data.git data/songys_data
```

다른 사람이 저장소를 클론할 때:

```bash
git clone https://github.com/YOUR_USERNAME/chatbot_project.git
cd chatbot_project
git submodule update --init --recursive
cp data/chatbot_data/ChatbotData.csv data/raw/
```

### 2.2 .gitignore에 데이터 제외 (대용량 데이터 권장)

```
# .gitignore
data/raw/ChatbotData.csv
data/raw/*.csv
data/processed/*
models/ko.bin
*.pt
```

## 3. CI/CD 파이프라인에 데이터 다운로드 추가

### 3.1 run_pipeline.sh 개선

```bash
#!/bin/bash

# 데이터 다운로드 (있으면 스킵)
if [ ! -f "data/raw/ChatbotData.csv" ]; then
    echo "Downloading ChatbotData..."
    cd data/raw
    wget https://github.com/songys/Chatbot_data/raw/master/ChatbotData.csv
    cd ../..
fi

# 파이프라인 실행
python scripts/01_preprocess.py
python scripts/02_build_corpus.py
python scripts/03_augmentation.py
python scripts/04_train.py
```

### 3.2 GitHub Actions 워크플로우

`.github/workflows/download_data.yml`:

```yaml
name: Download and Process Data

on: [push, pull_request]

jobs:
  prepare-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Download ChatbotData
        run: |
          mkdir -p data/raw
          cd data/raw
          wget https://github.com/songys/Chatbot_data/raw/master/ChatbotData.csv
          cd ../..
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run preprocessing
        run: python scripts/01_preprocess.py
```

## 4. 데이터 공유 옵션 비교

| 방법 | 장점 | 단점 | 추천 |
|------|------|------|------|
| **공개 GitHub 저장소** | 관리 용이, CI/CD 통합 쉬움 | 개인 데이터는 부적합 | ✓ 공개 데이터 |
| **Git 서브모듈** | 의존성 명확, 버전 관리 | 복잡성 증가 | 데이터 저장소 분리 |
| **대용량 파일 (LFS)** | GitHub LFS로 대용량 파일 관리 | 추가 설정 필요 | 매우 큰 파일용 |
| **Google Drive / S3** | 개인 데이터 보호 | 접근 관리 필요 | 개인/민감 데이터 |
| **.gitignore + 다운로드 스크립트** | 간단, 저장소 크기 작음 | 자동화 덜함 | 일반적인 경우 |

## 5. ChatbotData.csv 형식 검증

데이터가 올바른 형식인지 확인:

```python
import pandas as pd

csv_path = 'data/raw/ChatbotData.csv'
df = pd.read_csv(csv_path)

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values: {df.isnull().sum()}")
print(df.head())

# 예상 형식:
# - 첫 번째 컬럼: 질문 (또는 "question")
# - 두 번째 컬럼: 답변 (또는 "answer")
# - 선택: 세 번째 컬럼: 의도/카테고리 (또는 "intent")
```

## 6. 권장 설정

### 최적 방식 (공개 데이터)

1. **공개 저장소 직접 다운로드:**
   ```bash
   cd data/raw
   wget https://github.com/songys/Chatbot_data/raw/master/ChatbotData.csv
   ```

2. **.gitignore에 추가:**
   ```
   data/raw/*.csv
   ```

3. **README에 다운로드 명령 기록:**
   ```markdown
   # 데이터 준비
   
   ```bash
   cd data/raw
   wget https://github.com/songys/Chatbot_data/raw/master/ChatbotData.csv
   ```
   ```

### 개인 저장소가 있는 경우

1. **개인 저장소 생성:**
   - GitHub에서 private 저장소 생성
   - ChatbotData.csv 업로드

2. **프로젝트에 통합:**
   ```bash
   git submodule add git@github.com:YOUR_USERNAME/chatbot_data.git data/chatbot_data
   ```

3. **배포:**
   ```bash
   git clone --recurse-submodules https://github.com/YOUR_USERNAME/chatbot_project.git
   ```

## 7. 문제 해결

### Q: CSV 파일을 다운로드할 수 없습니다

A: 인터넷 연결 확인 또는 대체 방법 사용:
```bash
# curl 사용
curl -L -O https://github.com/songys/Chatbot_data/raw/master/ChatbotData.csv

# Python 사용
python -c "
import urllib.request
url = 'https://github.com/songys/Chatbot_data/raw/master/ChatbotData.csv'
urllib.request.urlretrieve(url, 'ChatbotData.csv')
"
```

### Q: Private 저장소에 접근할 수 없습니다

A: GitHub Personal Access Token 사용:
```bash
git clone https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/chatbot_data.git
```

## 8. RunPod 환경에서 권장 설정

RunPod remote server에서는 다음을 권장합니다:

```bash
# 설정 1: 시작 스크립트에 다운로드 포함
cat > setup.sh << 'EOF'
#!/bin/bash
cd /workspace/chatbot_project/data/raw

if [ ! -f ChatbotData.csv ]; then
    echo "Downloading ChatbotData..."
    wget https://github.com/songys/Chatbot_data/raw/master/ChatbotData.csv
fi

echo "Data ready!"
EOF

chmod +x setup.sh
./setup.sh
```

## 참고 링크

- **songys/Chatbot_data**: https://github.com/songys/Chatbot_data
- **GitHub Submodules**: https://git-scm.com/book/en/v2/Git-Tools-Submodules
- **GitHub Actions**: https://github.com/features/actions
- **Git LFS**: https://git-lfs.github.com/
