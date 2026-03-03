"""
Step 5-6: 트랜스포머 모델 학습
- <start>, <end> 토큰 추가
- 데이터 벡터화
- RTX 4090 최적화 배치 사이즈 설정
- 트랜스포머 모델 정의 및 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

class ChatbotDataset(Dataset):
    """챗봇 데이터셋 클래스"""
    
    def __init__(self, df, vocab, max_length=config.MAX_SEQ_LENGTH):
        """
        초기화
        
        Args:
            df (pd.DataFrame): 데이터
            vocab (dict): word2idx 매핑
            max_length (int): 최대 시퀀스 길이
        """
        self.questions = df['question'].values
        self.answers = df['answer'].values
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # 토큰화
        q_tokens = question.split()
        a_tokens = answer.split()
        
        # 인덱싱
        q_indices = self._tokens_to_indices(q_tokens)
        a_indices = self._tokens_to_indices(a_tokens)
        
        # <start>, <end> 토큰 추가
        a_indices_with_tokens = [self.vocab.get('<start>', 1)] + a_indices + [self.vocab.get('<end>', 1)]
        
        # 길이 조정 (패딩 또는 자르기)
        q_indices = self._pad_or_trim(q_indices, self.max_length)
        a_indices_with_tokens = self._pad_or_trim(a_indices_with_tokens, self.max_length)
        
        return {
            'question': torch.tensor(q_indices, dtype=torch.long),
            'answer': torch.tensor(a_indices_with_tokens, dtype=torch.long),
            'q_text': question,
            'a_text': answer
        }
    
    def _tokens_to_indices(self, tokens):
        """토큰을 인덱스로 변환"""
        indices = []
        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab[token])
            else:
                indices.append(self.vocab.get('<unk>', 1))
        return indices
    
    def _pad_or_trim(self, sequence, max_length):
        """시퀀스 패딩 또는 자르기"""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = [self.vocab.get('<pad>', 0)] * (max_length - len(sequence))
            return sequence + padding

class TransformerChatbot(nn.Module):
    """트랜스포머 기반 챗봇 모델"""
    
    def __init__(self, vocab_size, d_model=config.TRANSFORMER_D_MODEL, 
                 nhead=config.TRANSFORMER_NHEAD, num_layers=config.TRANSFORMER_NUM_LAYERS,
                 dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD, dropout=config.DROPOUT_RATE):
        """
        초기화
        
        Args:
            vocab_size (int): 어휘 크기
            d_model (int): 모델 차원
            nhead (int): 어텐션 헤드 개수
            num_layers (int): 트랜스포머 층 개수
            dim_feedforward (int): FFN 차원
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = self._get_positional_encoding(d_model, config.MAX_SEQ_LENGTH)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def _get_positional_encoding(self, d_model, max_len):
        """위치 인코딩 생성"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        return pe
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass
        
        Args:
            src (Tensor): 인코더 입력 (batch_size, seq_len)
            tgt (Tensor): 디코더 입력 (batch_size, seq_len)
            src_key_padding_mask (Tensor): 소스 패딩 마스크
            tgt_key_padding_mask (Tensor): 타겟 패딩 마스크
            
        Returns:
            Tensor: 출력 (batch_size, seq_len, vocab_size)
        """
        # 임베딩
        src_emb = self.embedding(src) * np.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * np.sqrt(self.d_model)
        
        # 위치 인코딩 추가
        src_emb = src_emb + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt_emb = tgt_emb + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        
        # 드롭아웃
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # 인코더
        encoder_output = self.transformer_encoder(
            src_emb, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 디코더
        decoder_output = self.transformer_decoder(
            tgt_emb,
            encoder_output,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 출력 계층
        output = self.fc_out(decoder_output)
        
        return output

class Tokenizer:
    """토크나이저 클래스"""
    
    def __init__(self, vocab):
        self.word2idx = vocab
        self.idx2word = {v: k for k, v in vocab.items()}
    
    def save(self, path):
        """토크나이저 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)
        print(f"✓ Tokenizer saved to {path}")

def create_vocab_from_corpus(corpus_path):
    """코퍼스에서 어휘 생성"""
    with open(corpus_path, 'rb') as f:
        corpus_data = pickle.load(f)
    
    return corpus_data['word2idx']

def train_epoch(model, dataloader, optimizer, criterion, device):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # 데이터 로드
        src = batch['question'].to(device)
        tgt = batch['answer'].to(device)
        
        # 패딩 마스크 생성
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)
        
        # Forward pass
        output = model(src, tgt, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        
        # loss 계산 (패딩 토큰 제외)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['question'].to(device)
            tgt = batch['answer'].to(device)
            
            src_padding_mask = (src == 0)
            tgt_padding_mask = (tgt == 0)
            
            output = model(src, tgt, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Step 5-6: 트랜스포머 모델 학습")
    print("=" * 60)
    
    # 데이터 로드
    if not os.path.exists(config.AUGMENTED_CSV):
        print(f"✗ Error: Augmented data not found")
        print("Please run 03_augmentation.py first")
        return
    
    print(f"\nLoading augmented data from {config.AUGMENTED_CSV}...")
    df = pd.read_csv(config.AUGMENTED_CSV)
    print(f"✓ Loaded {len(df)} sentences")
    
    # 어휘 로드
    if not os.path.exists(config.CORPUS_PKL):
        print(f"✗ Error: Corpus not found")
        print("Please run 02_build_corpus.py first")
        return
    
    print(f"Loading vocabulary from corpus...")
    vocab = create_vocab_from_corpus(config.CORPUS_PKL)
    print(f"✓ Vocabulary size: {len(vocab)}")
    
    # 데이터셋 및 데이터로더 생성
    print(f"\nCreating dataset...")
    dataset = ChatbotDataset(df, vocab)
    
    # 훈련/검증 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"✓ Train set size: {len(train_dataset)}")
    print(f"✓ Validation set size: {len(val_dataset)}")
    
    # RTX 4090 최적화 배치 사이즈
    # 4090은 24GB VRAM을 가지고 있으므로 큰 배치 사이즈 사용 가능
    batch_size = config.BATCH_SIZE  # 128
    
    # RTX 4090에서 메모리 사용량 추정
    num_params = (config.VOCAB_SIZE * config.TRANSFORMER_D_MODEL + 
                  config.MAX_SEQ_LENGTH * config.TRANSFORMER_D_MODEL * config.TRANSFORMER_NHEAD * batch_size)
    estimated_memory_mb = (num_params * 4) / (1024 * 1024)  # float32 = 4 bytes
    print(f"\nRTX 4090 최적화 설정:")
    print(f"Batch size: {batch_size}")
    print(f"Estimated memory per batch: ~{estimated_memory_mb:.0f}MB")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 모델 생성
    print(f"\nCreating model...")
    model = TransformerChatbot(vocab_size=len(vocab)).to(device)
    
    # 모델 파라미터 수
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {num_params:,}")
    
    # 옵티마이저와 loss function
    # optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 토큰(0) 무시
    # warmup + linear scheduler
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy='linear'
    )
    
    # 학습 루프
    print(f"\nStarting training...")
    print(f"Epochs: {config.EPOCHS}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config.EPOCHS):
        print(f"\n[Epoch {epoch + 1}/{config.EPOCHS}]")
        
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # 검증
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"✓ Best model saved (val_loss: {val_loss:.4f})")
    
    # 결과 저장
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'vocab_size': len(vocab),
        'model_params': num_params,
        'batch_size': batch_size,
        'epochs': config.EPOCHS,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(config.MODEL_PATH, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 토크나이저 저장
    tokenizer = Tokenizer(vocab)
    tokenizer.save(config.TOKENIZER_PATH)
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Model saved to {config.MODEL_SAVE_PATH}")
    print(f"✓ Results saved to {results_path}")

if __name__ == "__main__":
    main()
