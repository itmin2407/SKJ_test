#!/bin/bash

# 한국어 챗봇 프로젝트 전체 파이프라인 실행 스크립트

set -e

echo "=========================================="
echo "한국어 챗봇 프로젝트 파이프라인 시작"
echo "=========================================="

# 현재 디렉토리 확인
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 로그 디렉토리 생성
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 프로젝트 시작" > "$LOG_DIR/pipeline.log"

# Step 1-2: 데이터 전처리
echo ""
echo "=========================================="
echo "Step 1-2: 데이터 전처리"
echo "=========================================="

if [ -f "data/raw/ChatbotData.csv" ]; then
    python scripts/01_preprocess.py 2>&1 | tee -a "$LOG_DIR/preprocess.log"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step 1-2 완료" >> "$LOG_DIR/pipeline.log"
else
    echo "✗ Error: data/raw/ChatbotData.csv 파일을 찾을 수 없습니다."
    echo "Please place ChatbotData.csv in data/raw/ directory"
    exit 1
fi

# Step 3: Mecab 코퍼스 구축
echo ""
echo "=========================================="
echo "Step 3: Mecab 코퍼스 구축"
echo "=========================================="

python scripts/02_build_corpus.py 2>&1 | tee -a "$LOG_DIR/corpus.log"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step 3 완료" >> "$LOG_DIR/pipeline.log"

# Step 4: 데이터 증강
echo ""
echo "=========================================="
echo "Step 4: 데이터 증강"
echo "=========================================="

echo "Note: augmentation uses ko.bin; loader will retry binary/text/FastText formats, may skip augmentation if the model is huge or USE_AUGMENTATION=False."
python scripts/03_augmentation.py 2>&1 | tee -a "$LOG_DIR/augmentation.log"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step 4 완료" >> "$LOG_DIR/pipeline.log"

# Step 5-6: 모델 학습
echo ""
echo "=========================================="
echo "Step 5-6: 트랜스포머 모델 학습"
echo "=========================================="

python scripts/04_train.py 2>&1 | tee -a "$LOG_DIR/training.log"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step 5-6 완료" >> "$LOG_DIR/pipeline.log"

echo ""
echo "=========================================="
echo "✓ 파이프라인 완료!"
echo "=========================================="
echo ""
echo "생성된 파일:"
echo "- 정제된 데이터: data/processed/cleaned_data.csv"
echo "- 코퍼스: data/processed/corpus.pkl"
echo "- 증강된 데이터: data/processed/augmented_data.csv"
echo "- 학습된 모델: models/chatbot_model.pt"
echo "- 토크나이저: models/tokenizer.pkl"
echo "- 학습 결과: models/training_results.json"
echo ""
echo "로그 파일: logs/"
echo "=========================================="
