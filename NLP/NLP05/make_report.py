"""
NLP05 실험 결과를 RLHF_testNN.ipynb로 자동 생성하는 스크립트
RunPod에서 실행: python /workspace/test/NLP/NLP05/make_report.py
"""
import sys, os, glob, json
sys.path.insert(0, '/workspace/test/NLP/NLP05')

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# ── 경로 설정 ────────────────────────────────────────────────
RESULTS_DIR = '/workspace/test/NLP/NLP05/results'
NLP05_DIR   = '/workspace/test/NLP/NLP05'

existing = glob.glob(f'{NLP05_DIR}/RLHF_test*.ipynb')
next_num = len(existing) + 1
NOTEBOOK_NAME = f'RLHF_test{next_num:02d}'
NOTEBOOK_PATH = f'{NLP05_DIR}/{NOTEBOOK_NAME}.ipynb'

# ── 결과 파일 로드 (없으면 빈 데이터로 진행) ─────────────────
def load_json(filename, default=None):
    path = f'{RESULTS_DIR}/{filename}'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"⚠️  {filename} 없음 → 기본값 사용")
    return default if default is not None else []

sft_log     = load_json('SFT_experiments_log.json', [])
rm_log      = load_json('RM_experiments_log.json',  [])
ppo_log     = load_json('PPO_experiments_log.json', [])
comparisons = load_json('model_comparison.json',    [])
decodings   = load_json('decoding_results.json',    [])
quant       = load_json('quant_summary.json',       {})

# ── 헬퍼 함수 ────────────────────────────────────────────────
def truncate(text, maxlen=60):
    """텍스트를 maxlen으로 자르고 개행 제거"""
    if not text:
        return '(없음)'
    t = str(text).replace('\n', ' ')
    return t[:maxlen] + '...' if len(t) > maxlen else t

def build_training_table(log, stage):
    """experiments_log.json → 학습 결과 마크다운 표"""
    if not log:
        return f"*{stage}_experiments_log.json 없음 — 학습 미실행*"

    rec = log[-1]  # 가장 최근 실험
    history = rec.get('history', [])
    cfg = rec.get('cfg', {})

    rows = f"| 항목 | 값 |\n|------|-----|\n"
    rows += f"| Best epoch | {rec.get('best_epoch', '-')} |\n"
    rows += f"| 소요 시간 | {rec.get('duration_s', '-')}초 |\n"

    for k, v in cfg.items():
        rows += f"| cfg.{k} | {v} |\n"

    rows += "\n**Epoch별 loss:**\n\n"
    rows += "| Epoch | train_loss | val_loss |\n|-------|------------|----------|\n"
    for h in history:
        tl = f"{h['train_loss']:.4f}" if 'train_loss' in h else '-'
        vl = f"{h['val_loss']:.4f}" if h.get('val_loss') is not None else '-'
        rows += f"| {h.get('epoch', '-')} | {tl} | {vl} |\n"

    return rows

# ── 셀 구성 ──────────────────────────────────────────────────
cells = []

# ① 타이틀
cells.append(new_markdown_cell(
    "# NLP05 — KoChatGPT 업그레이드\n"
    "KoGPT2 기반 RLHF 3단계 파이프라인: SFT → RM → PPO\n\n"
    "**루브릭**\n"
    "- ① baseline vs SFT 비교\n"
    "- ② SFT vs RM/PPO 비교\n"
    "- ③ Decoding 실험 + LLM-as-a-Judge"))

# ② 환경 설정
cells.append(new_code_cell(
    "import os\n"
    "os.chdir('/workspace/test/NLP/NLP05')\n"
    "import sys\n"
    "sys.path.insert(0, '/workspace/test/NLP/NLP05')\n"
    "print('Working dir:', os.getcwd())"))

# ③ 패키지 설치
cells.append(new_code_cell(
    "import subprocess\n"
    "subprocess.run(['pip', 'install', '-q', 'transformers', 'torch', 'nltk',\n"
    "                'rouge-score', 'matplotlib', '--break-system-packages'])\n"
    "import nltk\n"
    "nltk.download('punkt', quiet=True)\n"
    "print('설치 완료')"))

# ④ STEP 1 — SFT
cells.append(new_markdown_cell("## STEP 1 — SFT (지도 미세조정)"))
cells.append(new_markdown_cell(f"### 학습 결과\n\n{build_training_table(sft_log, 'SFT')}"))

# ⑤ STEP 2 — RM
cells.append(new_markdown_cell("## STEP 2 — RM (보상 모델 학습)"))
cells.append(new_markdown_cell(f"### 학습 결과\n\n{build_training_table(rm_log, 'RM')}"))

# ⑥ STEP 3 — PPO
cells.append(new_markdown_cell("## STEP 3 — PPO (강화학습)"))
cells.append(new_markdown_cell(f"### 학습 결과\n\n{build_training_table(ppo_log, 'PPO')}"))

# ⑦ STEP 4 — baseline vs SFT vs PPO 비교
cells.append(new_markdown_cell("## STEP 4 — 모델 비교 평가"))

# ① baseline vs SFT
cells.append(new_markdown_cell("### ① baseline vs SFT 비교"))
if comparisons:
    rows = "| 프롬프트 | baseline | SFT |\n|----------|----------|-----|\n"
    for item in comparisons[:5]:
        rows += (f"| {truncate(item.get('prompt',''), 30)} "
                 f"| {truncate(item.get('baseline',''))} "
                 f"| {truncate(item.get('sft',''))} |\n")
    cells.append(new_markdown_cell(rows))
else:
    cells.append(new_markdown_cell("*model_comparison.json 없음*"))

# ② SFT vs PPO
cells.append(new_markdown_cell("### ② SFT vs PPO 비교"))
if comparisons:
    rows = "| 프롬프트 | SFT | PPO |\n|----------|-----|-----|\n"
    for item in comparisons[:5]:
        rows += (f"| {truncate(item.get('prompt',''), 30)} "
                 f"| {truncate(item.get('sft',''), 50)} "
                 f"| {truncate(item.get('ppo',''), 50)} |\n")
    cells.append(new_markdown_cell(rows))
else:
    cells.append(new_markdown_cell("*model_comparison.json 없음*"))

# ⑧ STEP 5 — 정량 평가 시각화
cells.append(new_markdown_cell("## STEP 5 — 정량 평가 (BLEU + ROUGE-L)"))

# 정량 수치 마크다운 표
if quant:
    bleu_b = quant.get('baseline', {}).get('avg_bleu', quant.get('baseline', {}).get('bleu', 0))
    bleu_s = quant.get('sft', {}).get('avg_bleu', quant.get('sft', {}).get('bleu', 0))
    bleu_p = quant.get('ppo', {}).get('avg_bleu', quant.get('ppo', {}).get('bleu', 0))
    rouge_b = quant.get('baseline', {}).get('avg_rouge_l', quant.get('baseline', {}).get('rouge_l', 0))
    rouge_s = quant.get('sft', {}).get('avg_rouge_l', quant.get('sft', {}).get('rouge_l', 0))
    rouge_p = quant.get('ppo', {}).get('avg_rouge_l', quant.get('ppo', {}).get('rouge_l', 0))

    cells.append(new_markdown_cell(
        f"| 모델 | BLEU | ROUGE-L |\n"
        f"|------|------|---------|  \n"
        f"| baseline | {bleu_b:.4f} | {rouge_b:.4f} |\n"
        f"| SFT | {bleu_s:.4f} | {rouge_s:.4f} |\n"
        f"| PPO | {bleu_p:.4f} | {rouge_p:.4f} |"))
else:
    bleu_b = bleu_s = bleu_p = 0
    rouge_b = rouge_s = rouge_p = 0
    cells.append(new_markdown_cell("*quant_summary.json 없음*"))

# 시각화 코드셀
cells.append(new_code_cell(
    "import json, matplotlib.pyplot as plt\n"
    "import config as cfg\n\n"
    "with open(f'{cfg.RESULTS_DIR}/quant_summary.json') as f:\n"
    "    quant = json.load(f)\n\n"
    "stages = list(quant.keys())\n"
    "bleu   = [quant[s].get('avg_bleu', quant[s].get('bleu', 0)) for s in stages]\n"
    "rouge  = [quant[s].get('avg_rouge_l', quant[s].get('rouge_l', 0)) for s in stages]\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
    "axes[0].bar(stages, bleu,  color=['#aec6cf','#779ecb','#4a7ebf'])\n"
    "axes[0].set_title('BLEU Score')\n"
    "axes[0].set_ylabel('BLEU')\n"
    "axes[1].bar(stages, rouge, color=['#aec6cf','#779ecb','#4a7ebf'])\n"
    "axes[1].set_title('ROUGE-L Score')\n"
    "axes[1].set_ylabel('ROUGE-L')\n"
    "plt.suptitle('baseline vs SFT vs PPO')\n"
    "plt.tight_layout()\n"
    "plt.savefig(f'{cfg.RESULTS_DIR}/quantitative_comparison.png', dpi=150)\n"
    "plt.show()"))

# ⑨ Decoding 실험
cells.append(new_markdown_cell("## Decoding 실험 (PPO 모델)"))
if decodings:
    strategies = ['greedy', 'beam_2', 'beam_4', 'topk_10', 'topk_50', 'topp_09']
    for i, item in enumerate(decodings[:3]):
        rows = f"### 샘플 {i+1}: {truncate(item.get('prompt',''), 40)}\n\n"
        rows += "| 전략 | 출력 |\n|------|------|\n"
        for s in strategies:
            rows += f"| {s} | {truncate(item.get(s, ''), 80)} |\n"
        cells.append(new_markdown_cell(rows))
else:
    cells.append(new_markdown_cell("*decoding_results.json 없음*"))

# ⑩ LLM-as-a-Judge
cells.append(new_markdown_cell("## LLM-as-a-Judge (SFT vs PPO)"))
if comparisons:
    for i, item in enumerate(comparisons[:3]):
        prompt = truncate(item.get('prompt', ''), 100)
        sft_out = truncate(item.get('sft', ''), 200)
        ppo_out = truncate(item.get('ppo', ''), 200)
        cells.append(new_markdown_cell(
            f"### 샘플 {i+1}\n\n"
            f"**[질문]** {prompt}\n\n"
            f"**[응답 A - SFT]** {sft_out}\n\n"
            f"**[응답 B - PPO]** {ppo_out}\n\n"
            "**평가 기준:** 유창성 / 관련성 / 정보성 (각 1~5점)"))
else:
    cells.append(new_markdown_cell("*model_comparison.json 없음*"))

# ⑪ 결론
cells.append(new_markdown_cell("## 결론 및 고찰"))

# SFT best loss 추출
sft_best = '-'
if sft_log and sft_log[-1].get('history'):
    sft_best = f"{min(h['train_loss'] for h in sft_log[-1]['history']):.4f}"

# RM best loss 추출
rm_best = '-'
if rm_log and rm_log[-1].get('history'):
    rm_best = f"{min(h['train_loss'] for h in rm_log[-1]['history']):.4f}"

# PPO best loss 추출
ppo_best = '-'
if ppo_log and ppo_log[-1].get('history'):
    ppo_best = f"{min(h['train_loss'] for h in ppo_log[-1]['history']):.4f}"

cells.append(new_markdown_cell(
    f"| 단계 | Best Loss | BLEU | ROUGE-L |\n"
    f"|------|-----------|------|---------|  \n"
    f"| SFT | {sft_best} | {bleu_s:.4f} | {rouge_s:.4f} |\n"
    f"| RM | {rm_best} | - | - |\n"
    f"| PPO | {ppo_best} | {bleu_p:.4f} | {rouge_p:.4f} |\n\n"
    "### RLHF 3단계 파이프라인 체험 요약\n\n"
    "| 단계 | 핵심 학습 |\n"
    "|------|----------|\n"
    "| SFT | 형식과 말투를 학습시킬 수 있다 |\n"
    "| RM | \"좋은 답\"을 점수로 정의할 수 있다 |\n"
    "| PPO | RM이 허술하면 Reward Hacking이 발생한다 |\n\n"
    "### 개선 방향\n"
    "1. `tokenizer.padding_side = 'left'` 적용\n"
    "2. KL penalty 계수 상향 (0.1 → 0.3~0.5)\n"
    "3. RM 학습 데이터 품질 개선\n"
    "4. PPO epochs 증가 또는 배치 크기 조정"))

# ── 노트북 저장 ──────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"✅ 노트북 저장 완료: {NOTEBOOK_PATH}")
print(f"   총 셀 수: {len(cells)}")
print(f"   파일명: {NOTEBOOK_NAME}.ipynb")
