import sys
sys.path.insert(0, '/workspace/test/NLP/NLP05')

import os, sys, json, random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup
)

# config는 RunPod 실행 시 절대경로로 import
sys.path.append("/workspace/NLP/NLP05")
import config as cfg
from scripts.experiment_tracker import ExperimentTracker

# ── 재현성 고정 ──────────────────────────────────────────
random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)

# ── 1. 데이터셋 ──────────────────────────────────────────
class SFTDataset(Dataset):
    """
    kochatgpt_1_SFT.jsonl 포맷: {"prompt": ..., "completion": ...}
    학습 목표: prompt + completion 전체를 causal LM으로 예측
    (GPT 계열은 다음 토큰 예측이므로 입력=출력=전체 시퀀스)
    """
    def __init__(self, path, tokenizer, max_len):
        self.samples = []
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            prompt = item["prompt"]
            completion = item["completion"]
            text = prompt + completion
            self.data.append(text)
        for text in self.data:
            enc  = tokenizer(text,
                             truncation=True,
                             max_length=max_len,
                             padding="max_length",
                             return_tensors="pt")
            input_ids = enc["input_ids"].squeeze()
            # GPT LM: labels = input_ids 그대로 (shift는 모델 내부에서 처리)
            self.samples.append({"input_ids": input_ids, "labels": input_ids.clone()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ── 2. 학습 루프 ─────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss    = outputs.loss          # CrossEntropyLoss (다음 토큰 예측)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# ── 3. 메인 ─────────────────────────────────────────────
def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 토크나이저 + 모델 로드 (HuggingFace pretrained)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.BASE_MODEL_NAME)
    tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                  'eos_token': '</s>',
                                  'bos_token': '<s>'})

    model = GPT2LMHeadModel.from_pretrained(cfg.BASE_MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 데이터셋
    dataset    = SFTDataset(cfg.SFT_DATA_PATH, tokenizer, cfg.SFT_MAX_LEN)
    loader     = DataLoader(dataset, batch_size=cfg.SFT_BATCH_SIZE, shuffle=True)
    print(f"SFT 데이터 수: {len(dataset)}")

    # 옵티마이저 + 스케줄러
    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.SFT_LR)
    total_steps = len(loader) * cfg.SFT_EPOCHS
    scheduler  = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.SFT_WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # 실험 추적
    tracker = ExperimentTracker(
        cfg={"epochs": cfg.SFT_EPOCHS, "batch_size": cfg.SFT_BATCH_SIZE,
             "lr": cfg.SFT_LR, "max_len": cfg.SFT_MAX_LEN},
        stage="SFT",
        results_dir=cfg.RESULTS_DIR
    )

    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, cfg.SFT_EPOCHS + 1):
        train_loss = train_epoch(model, loader, optimizer, scheduler, device)
        tracker.log_epoch(epoch, train_loss)

        if train_loss < best_loss:
            best_loss  = train_loss
            best_epoch = epoch
            os.makedirs(cfg.SFT_SAVE_PATH, exist_ok=True)
            model.save_pretrained(cfg.SFT_SAVE_PATH)
            tokenizer.save_pretrained(cfg.SFT_SAVE_PATH)
            print(f"  → 모델 저장: {cfg.SFT_SAVE_PATH}")

    tracker.save(best_epoch)
    print(f"\nSFT 완료. Best epoch={best_epoch}, Best loss={best_loss:.4f}")

if __name__ == "__main__":
    main()
