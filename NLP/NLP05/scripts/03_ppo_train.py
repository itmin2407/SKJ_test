import sys
sys.path.insert(0, '/workspace/test/NLP/NLP05')

import os, sys, json, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup
)

sys.path.append("/workspace/NLP/NLP05")
import config as cfg
from scripts.experiment_tracker import ExperimentTracker
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "rm_train", "/workspace/test/NLP/NLP05/scripts/02_rm_train.py")
rm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm_module)
RewardModel = rm_module.RewardModel

random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)


# ── 1. PPO 프롬프트 데이터셋 ─────────────────────────────
class PPODataset(Dataset):
    """
    kochatgpt_3_PPO.jsonl 포맷: {"prompt": ...}
    prompt만 있음 — 응답은 policy가 직접 생성
    """
    def __init__(self, path, tokenizer, max_len):
        self.prompts   = []
        self.tokenizer = tokenizer
        self.max_len   = max_len

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            self.prompts.append(item["prompt"])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            truncation=True,
            max_length=self.max_len // 2,   # 생성 공간 확보를 위해 절반만 사용
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
        }


# ── 2. KL Divergence 계산 ────────────────────────────────
def compute_kl(policy_logits, ref_logits, input_ids):
    """
    PPO에서 KL penalty: policy가 reference에서 너무 멀어지지 않도록 제약
    KL(policy || reference) = sum(policy * log(policy/reference))
    → 강화학습 전후 모델 분포를 유사하게 유지 (TRPO 원리)
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs    = F.log_softmax(ref_logits,    dim=-1)

    # 실제 생성된 토큰에 대한 log prob만 추출
    policy_lp = policy_log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    ref_lp     = ref_log_probs.gather(-1,    input_ids.unsqueeze(-1)).squeeze(-1)

    # KL = policy_lp - ref_lp (log space에서의 차이)
    kl = (policy_lp - ref_lp).mean()
    return kl


# ── 3. PPO Clipping Loss ─────────────────────────────────
def ppo_clip_loss(policy_logits, old_logits, input_ids, advantages):
    """
    TRPO 클리핑: 업데이트 크기를 1±ε 범위로 제한
    - 좋은 방향(advantage>0): 너무 많이 올리지 않음 → 차근차근 학습
    - 나쁜 방향(advantage<0): 무제한 페널티 → reward hacking 방지
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    old_log_probs    = F.log_softmax(old_logits,    dim=-1)

    policy_lp = policy_log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    old_lp     = old_log_probs.gather(-1,    input_ids.unsqueeze(-1)).squeeze(-1)

    # 확률 비율 r = exp(log_policy - log_old)
    ratio = torch.exp(policy_lp - old_lp)

    # 클리핑: r을 [1-ε, 1+ε] 범위로 제한
    eps     = cfg.PPO_CLIP_EPS
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)

    # PPO loss: min(ratio * advantage, clipped * advantage)의 음수
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss


# ── 4. 응답 생성 ─────────────────────────────────────────
def generate_response(model, input_ids, tokenizer, max_new_tokens=64):
    """
    Policy model로 응답 생성
    greedy decoding (PPO 학습 중에는 단순하게)
    """
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )
    return output


# ── 5. 메인 PPO 루프 ─────────────────────────────────────
def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.SFT_SAVE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Policy model: SFT 모델에서 시작, 업데이트됨
    policy_model = GPT2LMHeadModel.from_pretrained(cfg.SFT_SAVE_PATH).to(device)

    # Reference model: SFT 모델 고정, KL 계산용
    ref_model = GPT2LMHeadModel.from_pretrained(cfg.SFT_SAVE_PATH).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False   # reference는 업데이트 안 함

    # Reward model 로드 (RM 학습 완료 후 실행)
    reward_model = RewardModel(cfg.SFT_SAVE_PATH).to(device)
    reward_model.load_state_dict(
        torch.load(os.path.join(cfg.RM_SAVE_PATH, "rm_model.pt"), map_location=device)
    )
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False   # RM도 업데이트 안 함

    dataset = PPODataset(cfg.PPO_DATA_PATH, tokenizer, cfg.PPO_MAX_LEN)
    loader  = DataLoader(dataset, batch_size=cfg.PPO_BATCH_SIZE, shuffle=True)
    print(f"PPO 데이터 수: {len(dataset)}")

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.PPO_LR)
    total_steps = len(loader) * cfg.PPO_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=total_steps
    )

    tracker = ExperimentTracker(
        cfg={"epochs": cfg.PPO_EPOCHS, "batch_size": cfg.PPO_BATCH_SIZE,
             "lr": cfg.PPO_LR, "kl_coef": cfg.PPO_KL_COEF,
             "clip_eps": cfg.PPO_CLIP_EPS},
        stage="PPO",
        results_dir=cfg.RESULTS_DIR
    )

    best_loss, best_epoch = float("inf"), 0

    for epoch in range(1, cfg.PPO_EPOCHS + 1):
        policy_model.train()
        total_loss, total_reward, total_kl = 0, 0, 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            # ① 응답 생성 (policy)
            generated = generate_response(policy_model, input_ids, tokenizer)

            # ② RM으로 reward 계산
            with torch.no_grad():
                rm_score = reward_model(generated, attention_mask=None)

            # ③ KL divergence 계산
            policy_logits = policy_model(generated).logits[:, :-1]
            with torch.no_grad():
                ref_logits = ref_model(generated).logits[:, :-1]
            response_ids = generated[:, 1:]   # shift

            kl = compute_kl(policy_logits, ref_logits, response_ids)

            # ④ total reward = RM reward - KL penalty
            total_r = rm_score - cfg.PPO_KL_COEF * kl.detach()

            # advantage: 배치 내 상대적 점수 정규화
            advantages = (total_r - total_r.mean()) / (total_r.std() + 1e-8)
            advantages = advantages.unsqueeze(1).expand_as(response_ids).float()

            # ⑤ PPO clipping loss
            with torch.no_grad():
                old_logits = ref_logits   # 첫 업데이트는 ref = old policy
            loss = ppo_clip_loss(policy_logits, old_logits, response_ids, advantages)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss   += loss.item()
            total_reward += rm_score.mean().item()
            total_kl     += kl.item()

        avg_loss   = total_loss   / len(loader)
        avg_reward = total_reward / len(loader)
        avg_kl     = total_kl     / len(loader)

        tracker.log_epoch(epoch, avg_loss)
        print(f"  avg_reward={avg_reward:.4f} | avg_kl={avg_kl:.4f}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_epoch = epoch
            os.makedirs(cfg.PPO_SAVE_PATH, exist_ok=True)
            policy_model.save_pretrained(cfg.PPO_SAVE_PATH)
            tokenizer.save_pretrained(cfg.PPO_SAVE_PATH)
            print(f"  → 모델 저장: {cfg.PPO_SAVE_PATH}")

    tracker.save(best_epoch)
    print(f"\nPPO 완료. Best epoch={best_epoch}, Best loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
