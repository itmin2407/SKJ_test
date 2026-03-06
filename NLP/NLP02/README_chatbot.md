# 한국어 챗봇 프로젝트 (RTX 4090 기반)

트랜스포머 기반 Seq2Seq 모델을 사용하여 **한국어 챗봇을 학습하는 프로젝트**입니다.
RTX 4090 GPU 환경을 기준으로 학습 파이프라인이 구성되어 있습니다.

---

# 프로젝트 개요

이 프로젝트는 다음과 같은 단계로 구성됩니다.

| 단계       | 설명                 | 스크립트                 |
| -------- | ------------------ | -------------------- |
| Step 1–2 | 데이터 전처리            | `01_preprocess.py`   |
| Step 3   | Mecab 기반 코퍼스 구축    | `02_build_corpus.py` |
| Step 4   | Word2Vec 기반 데이터 증강 | `03_augmentation.py` |
| Step 5–6 | Transformer 모델 학습  | `04_train.py`        |

---

# 시스템 요구사항

* GPU: **RTX 4090 (24GB VRAM)**
* Python: **3.10+**
* CUDA: **12.1+**
* Mecab 설치 필요

---

# 프로젝트 구조

```
chatbot_project/
│
├── data/
│   ├── raw/
│   │   └── ChatbotData.csv
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       ├── augmented_data.csv
│       └── corpus.pkl
│
├── models/
│   ├── ko.bin
│   ├── chatbot_model.pt
│   ├── tokenizer.pkl
│   └── training_results.json
│
├── scripts/
│   ├── 01_preprocess.py
│   ├── 02_build_corpus.py
│   ├── 03_augmentation.py
│   └── 04_train.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── config.py
├── requirements.txt
└── README.md
```

---

# 설치 방법

## 1. 저장소 클론

```
git clone <repository_url>
cd chatbot_project
```

---

## 2. Python 가상환경 생성

```
python -m venv venv
source venv/bin/activate
```

Windows

```
venv\Scripts\activate
```

---

## 3. PyTorch 설치 (CUDA 12.1)

```
pip install torch torchvision torchaudio \
--index-url https://download.pytorch.org/whl/cu121
```

---

## 4. 의존성 설치

```
pip install -r requirements.txt
```

---

## 5. Mecab 설치

### Ubuntu / Debian

```
sudo apt-get update
sudo apt-get install -y mecab mecab-ko mecab-ko-dic
```

### Conda

```
conda install -c conda-forge mecab mecab-ko-dic
```

---

# Word2Vec 모델 다운로드

한국어 Word2Vec 모델을 다운로드합니다.

### 옵션 1

```
cd models

wget https://github.com/Kyubyong/wordvectors/releases/download/korean/ko.bin

cd ..
```

### 옵션 2 (FastText)

```
cd models

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz

gunzip cc.ko.300.bin.gz
mv cc.ko.300.bin ko.bin

cd ..
```

---

# 데이터 준비

챗봇 데이터 CSV 파일을 준비합니다.

파일 위치

```
data/raw/ChatbotData.csv
```

CSV 형식

```
question,answer
안녕하세요,안녕하세요! 어떻게 도와드릴까요?
날씨가 어떻게 되나요?,날씨 정보는 현재 제공하고 있지 않습니다.
```

---

# 실행 방법

## Step별 실행

```
python scripts/01_preprocess.py
python scripts/02_build_corpus.py
python scripts/03_augmentation.py
python scripts/04_train.py
```

---

## 전체 파이프라인 실행

```
bash run_pipeline.sh
```

---

# RTX 4090 최적화 설정

이 프로젝트는 RTX 4090 GPU 기준으로 다음 설정이 적용되어 있습니다.

| 항목              | 값    |
| --------------- | ---- |
| Batch Size      | 128  |
| d_model         | 768  |
| num_heads       | 12   |
| num_layers      | 6    |
| dim_feedforward | 3072 |

예상 GPU 메모리 사용량

```
약 15 ~ 18GB VRAM
```

---

# 주요 설정 파일

`config.py`에서 주요 학습 파라미터를 수정할 수 있습니다.

```
VOCAB_SIZE = 10000
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 50

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
```

---

# 출력 파일

| 단계       | 파일                    | 설명            |
| -------- | --------------------- | ------------- |
| Step 1–2 | cleaned_data.csv      | 정제된 데이터       |
| Step 3   | corpus.pkl            | Mecab 토큰화 데이터 |
| Step 4   | augmented_data.csv    | 증강 데이터        |
| Step 5–6 | chatbot_model.pt      | 학습된 모델        |
| Step 5–6 | tokenizer.pkl         | 토크나이저         |
| Step 5–6 | training_results.json | 학습 결과         |

---

# 학습 모니터링

실시간 로그 확인

```
tail -f logs/training.log
```

학습 곡선 분석

```
jupyter notebook notebooks/analysis.ipynb
```

---

# 문제 해결

## Mecab 설치 오류

```
sudo apt-get install -y libmecab-dev mecab mecab-ko mecab-ko-dic
```

---

## Word2Vec 모델 오류

확인 사항

* `models/ko.bin` 존재 여부
* 파일 크기 **1GB 이상**
* 손상 시 재다운로드

---

## GPU 메모리 부족

다음 값을 줄입니다.

```
BATCH_SIZE
TRANSFORMER_D_MODEL
MAX_SEQ_LENGTH
```

---

## CUDA 확인

```
nvcc --version
```

```
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 예상 성능 (RTX 4090)

| 작업                       | 성능                |
| ------------------------ | ----------------- |
| 데이터 처리                   | ~100K samples/sec |
| 모델 학습                    | ~500 samples/sec  |
| 20 epochs (100K samples) | 약 40분             |

---

# 라이선스

MIT License

---

# 참고 자료

* PyTorch
  https://pytorch.org/docs

* HuggingFace Transformers
  https://huggingface.co

* KoNLPy
  https://konlpy.org
