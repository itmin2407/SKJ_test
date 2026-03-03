"""
프로젝트 설정 파일
RTX 4090 (24GB VRAM) 최적화 설정
"""

import os

# 데이터 경로
DATA_RAW_PATH = "./data/raw"
DATA_PROCESSED_PATH = "./data/processed"
MODEL_PATH = "./models"

# 파일 경로
CSV_FILE = os.path.join(DATA_RAW_PATH, "ChatbotData.csv")
CLEANED_CSV = os.path.join(DATA_PROCESSED_PATH, "cleaned_data.csv")
AUGMENTED_CSV = os.path.join(DATA_PROCESSED_PATH, "augmented_data.csv")
CORPUS_PKL = os.path.join(DATA_PROCESSED_PATH, "corpus.pkl")
KO_BIN_MODEL = os.path.join(MODEL_PATH, "ko.bin")

# 모델 파라미터
VOCAB_SIZE = 10000
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 50

# 트레이닝 파라미터 (RTX 4090 최적화)
BATCH_SIZE = 128  # RTX 4090은 충분한 VRAM이 있으므로 큰 배치 사이즈 사용
EPOCHS = 20
LEARNING_RATE = 0.0005  # lowered from 0.001 to help more stable convergence
DROPOUT_RATE = 0.3

# 트랜스포머 모델 파라미터
# model size enlarged for RTX 4090
TRANSFORMER_D_MODEL = 768      # increased from 512 to 768
TRANSFORMER_NHEAD = 12         # more heads to match d_model
TRANSFORMER_NUM_LAYERS = 6     # deeper network
TRANSFORMER_DIM_FEEDFORWARD = 3072  # larger feed-forward network

# 데이터 증강 설정
AUGMENTATION_FACTOR = 3  # 데이터를 3배로 증강
# augmentation 단계 자체를 건너뛰려면 False로 설정
USE_AUGMENTATION = True

# 정규식 정제 설정
REGEX_PATTERNS = [
    (r'[^\w\s가-힣.!?]', ''),  # 특수문자 제거 (한글, 영문, 숫자, 마침표, 물음표, 느낌표만 유지)
    (r'\s+', ' '),              # 연속 공백 제거
    (r'^\s+|\s+$', ''),          # 앞뒤 공백 제거
]

# 모델 저장 경로
MODEL_SAVE_PATH = os.path.join(MODEL_PATH, "chatbot_model.pt")
TOKENIZER_PATH = os.path.join(MODEL_PATH, "tokenizer.pkl")

# GPU 설정
USE_GPU = True
DEVICE = "cuda" if USE_GPU else "cpu"

# 시드 설정 (재현성)
RANDOM_SEED = 42
