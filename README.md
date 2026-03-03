<<<<<<< HEAD
# AIFFEL_quest_eng

## Repository Structure

```bash
AIFFEL_quest_eng
├── Computer_Vision
│   ├── CV01
│   │   ├── README.md
│   │   └── .ipynb
│   ├── CV02
│   │   ├── README.md
│   │   └── .ipynb
│   └── CV03
│       ├── README.md
│       └── .ipynb
├── Data_Analysis
│   ├── DA01
│   │   ├── README.md
│   │   └── .ipynb
│   └── DA02
│       ├── README.md
│       └── .ipynb
├── Deployment
│   ├── Contents
│   │   ├── README.md
│   │   └── .ipynb
│   └── Final_Code
│       ├── README.md
│       └── .ipynb
├── LLM_Application
│   ├── LLM01
│   │   ├── README.md
│   │   └── .ipynb
│   ├── LLM02
│   │   ├── README.md
│   │   └── .ipynb
│   ├── LLM03
│   │   ├── README.md
│   │   └── .ipynb
│   ├── LLM04
│   │   ├── README.md
│   │   └── .ipynb
│   └── LLM05
│       ├── README.md
│       └── .ipynb
├── MLOps
│   ├── MLOps01
│   │   ├── README.md
│   │   └── .ipynb
│   ├── MLOps02
│   │   ├── README.md
│   │   └── .ipynb
│   ├── MLOps03
│   │   ├── README.md
│   │   └── .ipynb
│   ├── MLOps04
│   │   ├── README.md
│   │   └── .ipynb
│   ├── MLOps05
│   │   ├── README.md
│   │   └── .ipynb
│   ├── MLOps06
│   │   ├── README.md
│   │   └── .ipynb
│   └── MLOps07
│       ├── README.md
│       └── .ipynb
├── Main_Quest
│   ├── Quest01
│   │   ├── README.md
│   │   └── .ipynb
│   ├── Quest02
│   │   ├── README.md
│   │   └── .ipynb
│   ├── Quest03
│   │   ├── README.md
│   │   └── .ipynb
│   ├── Quest04
│   │   ├── README.md
│   │   └── .ipynb
│   └── Quest05
│       └── README.md
├── NLP
│   ├── NLP01
│   │   ├── README.md
│   │   └── .ipynb
│   ├── NLP02
│   │   ├── README.md
│   │   └── .ipynb
│   ├── NLP03
│   │   ├── README.md
│   │   └── .ipynb
│   ├── NLP04
│   │   ├── README.md
│   │   └── .ipynb
│   └── NLP05
│       ├── README.md
│       └── .ipynb
├── Python
│   ├── Py01
│   │   ├── README.md
│   │   └── .ipynb
│   ├── Py02
│   │   ├── README.md
│   │   └── .ipynb
│   ├── Py03
│   │   ├── README.md
│   │   └── .ipynb
│   └── Py04
│       ├── README.md
│       └── .ipynb
└── README.md
```
=======
# 한국어 챗봇 프로젝트

RTX 4090을 활용한 한국어 챗봇 개발 프로젝트입니다. 트랜스포머 기반 시퀀스-투-시퀀스 모델을 사용합니다.

## 프로젝트 개요

이 프로젝트는 다음 단계를 따릅니다:

1. **Step 1-2: 데이터 전처리** (`01_preprocess.py`)
   - CSV 데이터 로드
   - 정규식을 사용한 데이터 정제
   - 결측값 및 중복 제거

2. **Step 3: Mecab 코퍼스 구축** (`02_build_corpus.py`)
   - KoNLPy Mecab을 사용한 형태소 분석
   - 어휘 사전 구축
   - 토큰화된 데이터 저장

3. **Step 4: 데이터 증강** (`03_augmentation.py`)
   - Word2Vec (ko.bin) 기반 유사 단어 생성
   - 데이터를 3배로 증강

4. **Step 5-6: 모델 학습** (`04_train.py`)
   - Transformer 기반 Seq2Seq 모델
   - <start>, <end> 토큰 추가
   - RTX 4090 최적화 배치 사이즈 설정
   - 모델 학습 및 검증

## 시스템 요구사항

- GPU: RTX 4090 (24GB VRAM)
- Python: 3.10+
- CUDA: 12.1+
- Mecab 설치 필요

## 설치 방법

### 1. 저장소 클론 및 환경 설정

```bash
cd /workspace/chatbot_project
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는 venv\Scripts\activate (Windows)
```

### 2. 의존성 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Mecab 설치 (Linux)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y mecab mecab-ko mecab-ko-dic

# 또는 Conda를 사용할 경우
conda install -c conda-forge mecab mecab-ko-dic
```

### 4. 한국어 Word2Vec 모델 다운로드

ko.bin 모델을 다운로드하여 `models/` 폴더에 저장해야 합니다.

**옵션 1: Kyubyong의 wordvectors 다운로드**
```bash
cd models
wget https://github.com/Kyubyong/wordvectors/releases/download/korean/ko.bin
cd ..
```

**옵션 2: FastText CBOW 모델 사용**
```bash
cd models
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz
gunzip cc.ko.300.bin.gz
mv cc.ko.300.bin ko.bin
cd ..
```

### 5. 데이터 준비

챗봇 데이터 CSV 파일을 준비하세요:
- 파일명: `ChatbotData.csv`
- 위치: `data/raw/ChatbotData.csv`
- 형식: 최소 2개 컬럼 (question, answer)

```csv
question,answer
안녕하세요,안녕하세요! 어떻게 도와드릴까요?
날씨가 어떻게 되나요?,날씨 정보는 현재 제공하고 있지 않습니다.
```

## 실행 방법

### Step별 실행

> **Augmentation tip:**
> * `03_augmentation.py`가 `UnicodeDecodeError`를 내면 `ko.bin` 형식이 잘못되었을 수 있습니다. 스크립트가 바이너리/텍스트/FastText 형식을 자동으로 시도하므로 코드가 이전보다 안정적입니다.
> * 모델 로딩이 너무 오래 걸리거나 반응이 없으면, `ko.bin` 파일이 매우 커서 발생할 수 있습니다. 이 경우 스크립트는 자동으로 증강을 건너뛰고 원본 데이터를 사용합니다.
> * 완전히 증강 단계를 끄려면 `config.py`에서 `USE_AUGMENTATION = False`로 설정하세요.

```bash
# Step 1-2: 데이터 전처리
python scripts/01_preprocess.py

# Step 3: Mecab 코퍼스 구축
python scripts/02_build_corpus.py

# Step 4: 데이터 증강
python scripts/03_augmentation.py

# Step 5-6: 모델 학습
python scripts/04_train.py
```

### 전체 파이프라인 실행 (스크립트)

```bash
bash run_pipeline.sh
```

## RTX 4090 최적화 설정

이 프로젝트는 RTX 4090의 성능을 최대한 활용하도록 최적화되어 있습니다:

- **Batch Size**: 128 (RTX 4090의 24GB VRAM 최적화)
- **Model Size**:
  - d_model: 768 (was 512, increased for RTX 4090)
  - num_heads: 12
  - num_layers: 6
  - dim_feedforward: 3072
- **Mixed Precision**: 가능 (추가 최적화)
- **Memory Usage**: ~15-18GB per batch

배치 사이즈를 조정하려면 `config.py`의 `BATCH_SIZE`를 수정하세요.

## 설정 파일

`config.py`에서 다음을 커스터마이즈할 수 있습니다:

```python
# 데이터 경로
VOCAB_SIZE = 10000
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 50

# 트레이닝 파라미터
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3

# 트랜스포머 파라미터
TRANSFORMER_D_MODEL = 768  # increased for RTX 4090
TRANSFORMER_NHEAD = 12
TRANSFORMER_NUM_LAYERS = 6
TRANSFORMER_DIM_FEEDFORWARD = 3072
```

## 프로젝트 구조

```
chatbot_project/
├── data/
│   ├── raw/                    # 원본 데이터
│   │   └── ChatbotData.csv
│   └── processed/              # 처리된 데이터
│       ├── cleaned_data.csv
│       ├── augmented_data.csv
│       └── corpus.pkl
├── models/
│   ├── ko.bin                  # Word2Vec 모델
│   ├── chatbot_model.pt        # 학습된 모델
│   ├── tokenizer.pkl           # 토크나이저
│   └── training_results.json   # 학습 결과
├── scripts/
│   ├── 01_preprocess.py        # 전처리
│   ├── 02_build_corpus.py      # 코퍼스 구축
│   ├── 03_augmentation.py      # 데이터 증강
│   └── 04_train.py             # 모델 학습
├── notebooks/
│   └── analysis.ipynb          # 분석 및 시각화
├── config.py                   # 설정 파일
├── requirements.txt            # 의존성
└── README.md                   # 이 파일
```

## 출력 파일

각 단계별로 생성되는 파일:

| 단계 | 출력 파일 | 설명 |
|------|----------|------|
| 1-2 | `data/processed/cleaned_data.csv` | 정제된 데이터 |
| 3 | `data/processed/corpus.pkl` | Mecab 토큰화 결과 및 어휘사전 |
| 4 | `data/processed/augmented_data.csv` | 증강된 데이터 |
| 5-6 | `models/chatbot_model.pt` | 학습된 모델 |
| 5-6 | `models/tokenizer.pkl` | 토크나이저 |
| 5-6 | `models/training_results.json` | 학습 메트릭 |

## 모니터링

학습 진행상황을 모니터링하려면:

```bash
# 터미널에서 실시간 로그 확인
tail -f logs/training.log

# 학습 곡선 확인 (Jupyter Notebook)
jupyter notebook notebooks/analysis.ipynb
```

## 문제 해결

### Mecab 설치 오류

```bash
# Linux (Ubuntu/Debian)
sudo apt-get install -y libmecab-dev mecab mecab-ko mecab-ko-dic

# macOS
brew install mecab mecab-ko mecab-ko-dic

# Conda (크로스 플랫폼)
conda install -c conda-forge mecab mecab-ko-dic
```

### ko.bin 모델 로드 오류

- 모델이 `models/ko.bin`에 있는지 확인
- 모델 파일 크기 > 1GB 확인
- 파일이 손상되었을 경우 다시 다운로드

### Out of Memory 오류

- `config.py`에서 `BATCH_SIZE` 감소 (128 → 64 또는 32)
- `TRANSFORMER_D_MODEL` 감소 (512 → 256)
- `MAX_SEQ_LENGTH` 감소 (50 → 32)

### CUDA 오류

```bash
# CUDA 버전 확인
nvcc --version

# PyTorch CUDA 호환성 확인
python -c "import torch; print(torch.cuda.is_available())"
```

## Performance 예상치

RTX 4090에서의 예상 성능 (배치 사이즈: 128):

- Data 처리: ~100K samples/sec
- 모델 학습: ~500 samples/sec
- 20 epochs, 100K samples: ~40분

## 라이선스

MIT License

## 참고자료

- [PyTorch Documentation](https://pytorch.org/docs)
- [Transformers Library](https://huggingface.co/transformers)
- [KoNLPy Documentation](https://konlpy.org)
- [RTX 4090 CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)

## 문의

프로젝트 관련 질문이 있으시면 이슈를 등록해주세요.
>>>>>>> 9e320f7 (Upload chatbot project to NLP02/chatbot)
