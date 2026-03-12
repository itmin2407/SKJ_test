import sys
sys.path.insert(0, '/workspace/test/NLP/NLP05')

import os, sys, json, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Model, PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup
)

sys.path.append("/workspace/NLP/NLP05")
import config as cfg
from scripts.experiment_tracker import ExperimentTracker

random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)

# ── 1. Reward Model 아키텍처 ─────────────────────────────
class RewardModel(nn.Module):
    """
    KoGPT2 백본 위에 scalar 점수 헤드를 붙인 구조.
    입력: (prompt + response) 토큰 시퀀스
    출력: 1개의 실수 점수 (높을수록 좋은 응답)
    """
    def __init__(self, backbone_path):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(backbone_path)
        hidden_size   = self.backbone.config.hidden_size
        # 마지막 토큰의 hidden state → scalar 점수
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs    = self.backbone(input_ids=input_ids,
                                   attention_mask=attention_mask)
        # last token hidden state 사용 (GPT는 left-to-right이므로 마지막이 전체 맥락 포함)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        score       = self.score_head(last_hidden).squeeze(-1)  # (batch,)
        return score

# ── 2. 데이터셋 ──────────────────────────────────────────
class RMDataset(Dataset):
    """
    kochatgpt_2_RM.jsonl 포맷:
    {"prompt": ..., "chosen": ..., "rejected": ...}
    또는 {"prompt": ..., "completion_0": ..., "completion_1": ..., "completion_2": ...}
    → (chosen, rejected) 페어로 변환하여 pairwise ranking loss 적용
    """
    def __init__(self, path, tokenizer, max_len):
        self.samples   = []
        self.tokenizer = tokenizer
        self.max_len   = max_len

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
                prompt = item["prompt"]

                # 포맷 A: chosen/rejected 키가 있는 경우
                if "chosen" in item and "rejected" in item:
                    self.samples.append({
                        "chosen":   prompt + item["chosen"],
                        "rejected": prompt + item["rejected"]
                    })
                # 포맷 B: completion_0~N 키가 있는 경우 (첫 번째=best)
                else:
                    completions = [v for k, v in item.items()
                                   if k.startswith("completion")]
                    if len(completions) >= 2:
                        self.samples.append({
                            "chosen":   prompt + completions[0],
                            "rejected": prompt + completions[-1]
                        })

    def _encode(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s       = self.samples[idx]
        chosen  = self._encode(s["chosen"])
        rejected= self._encode(s["rejected"])
        return {
            "chosen_ids":       chosen["input_ids"].squeeze(),
            "chosen_mask":      chosen["attention_mask"].squeeze(),
            "rejected_ids":     rejected["input_ids"].squeeze(),
            "rejected_mask":    rejected["attention_mask"].squeeze(),
        }

# ── 3. Pairwise Ranking Loss ─────────────────────────────
def ranking_loss(chosen_score, rejected_score):
    """
    chosen 점수가 rejected보다 높아야 한다는 목표.
    loss = -log(sigmoid(chosen - rejected))
    → chosen - rejected 차이가 클수록 loss 감소
    """
    return -torch.nn.functional.logsigmoid(chosen_score - rejected_score).mean()

# ── 4. 학습 루프 ─────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        chosen_ids   = batch["chosen_ids"].to(device)
        chosen_mask  = batch["chosen_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_mask= batch["rejected_mask"].to(device)

        chosen_score  = model(chosen_ids,  chosen_mask)
        rejected_score= model(rejected_ids, rejected_mask)

        loss = ranking_loss(chosen_score, rejected_score)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        # 정확도: chosen 점수 > rejected 점수인 비율
        correct += (chosen_score > rejected_score).sum().item()
        total   += len(chosen_score)

    return total_loss / len(loader), correct / total

# ── 5. 메인 ─────────────────────────────────────────────
def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # SFT 완료 모델을 백본으로 사용 (SFT → RM 순서 준수)
    backbone_path = cfg.SFT_SAVE_PATH
    print(f"백본 로드: {backbone_path}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(backbone_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = RewardModel(backbone_path).to(device)

    dataset = RMDataset(cfg.RM_DATA_PATH, tokenizer, cfg.RM_MAX_LEN)
    loader  = DataLoader(dataset, batch_size=cfg.RM_BATCH_SIZE, shuffle=True)
    print(f"RM 데이터 수: {len(dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.RM_LR)
    total_steps = len(loader) * cfg.RM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=total_steps
    )

    tracker = ExperimentTracker(
        cfg={"epochs": cfg.RM_EPOCHS, "batch_size": cfg.RM_BATCH_SIZE,
             "lr": cfg.RM_LR, "max_len": cfg.RM_MAX_LEN},
        stage="RM",
        results_dir=cfg.RESULTS_DIR
    )

    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, cfg.RM_EPOCHS + 1):
        train_loss, acc = train_epoch(model, loader, optimizer, scheduler, device)
        tracker.log_epoch(epoch, train_loss)
        print(f"  Ranking Accuracy: {acc:.4f}")

        if train_loss < best_loss:
            best_loss  = train_loss
            best_epoch = epoch
            os.makedirs(cfg.RM_SAVE_PATH, exist_ok=True)
            # RM은 커스텀 모델이므로 state_dict로 저장
            torch.save(model.state_dict(),
                       os.path.join(cfg.RM_SAVE_PATH, "rm_model.pt"))
            tokenizer.save_pretrained(cfg.RM_SAVE_PATH)
            print(f"  → 모델 저장: {cfg.RM_SAVE_PATH}")

    tracker.save(best_epoch)
    print(f"\nRM 완료. Best epoch={best_epoch}, Best loss={best_loss:.4f}")

if __name__ == "__main__":
    main()
