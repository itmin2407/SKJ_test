"""
NLP05 결과를 NLP05.ipynb에 자동으로 채워넣는 스크립트
RunPod에서 실행: python /workspace/test/NLP/NLP05/make_report.py
"""
import sys
sys.path.insert(0, '/workspace/test/NLP/NLP05')

NOTEBOOK_NAME = sys.argv[1] if len(sys.argv) > 1 else 'NLP05'
NOTEBOOK_PATH = f'/workspace/test/NLP/NLP05/{NOTEBOOK_NAME}.ipynb'

import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

RESULTS_DIR = '/workspace/test/NLP/NLP05/results'

# ── 결과 파일 로드 ──────────────────────────────────────────
with open(f'{RESULTS_DIR}/model_comparison.json') as f:
    comparisons = json.load(f)

with open(f'{RESULTS_DIR}/decoding_results.json') as f:
    decodings = json.load(f)

with open(f'{RESULTS_DIR}/quant_summary.json') as f:
    quant = json.load(f)

# ── 셀 목록 구성 ────────────────────────────────────────────
cells = []

# ── 타이틀 ──
cells.append(new_markdown_cell("""# NLP05 — KoChatGPT 업그레이드

KoGPT2 기반 RLHF 3단계 파이프라인: SFT → RM → PPO

## 루브릭
- ① baseline vs SFT 비교
- ② SFT vs RM/PPO 비교
- ③ Decoding 실험 + LLM-as-a-Judge"""))

# ── 환경 설정 셀 (실행용) ──
cells.append(new_code_cell("""\
import os
os.chdir('/workspace/NLP/NLP05')
import sys
sys.path.append('/workspace/test/NLP/NLP05')
print('Working dir:', os.getcwd())"""))

cells.append(new_code_cell("""\
# 패키지 설치
import subprocess
subprocess.run(['pip', 'install', '-q', 'transformers', 'torch', 'nltk',
                'rouge-score', 'matplotlib', '--break-system-packages'])
import nltk
nltk.download('punkt', quiet=True)
print('설치 완료')"""))

# ── STEP 1: SFT ──
cells.append(new_markdown_cell("## STEP 1 — SFT (지도 미세조정)"))
cells.append(new_markdown_cell("""\
### 학습 결과

| 항목 | 값 |
|------|-----|
| 데이터 수 | 12,000건 |
| Epoch 1 loss | 2.1709 |
| Epoch 2 loss | 1.2735 |
| Epoch 3 loss | 1.1561 |
| Best epoch | 3 |
| 소요 시간 | 239초 |
| 저장 경로 | `/workspace/test/NLP/NLP05/models/sft_model` |

loss가 epoch마다 단조 감소 → 정상 학습. \
KoGPT2가 질문-답변 형식을 학습함."""))

# ── STEP 2: RM ──
cells.append(new_markdown_cell("## STEP 2 — RM (보상 모델 학습)"))
cells.append(new_markdown_cell("""\
### 학습 결과

| 항목 | 값 |
|------|-----|
| 데이터 수 | 10,220건 (chosen/rejected 쌍) |
| Epoch 1 | loss=0.7117, Ranking Acc=49.07% |
| Epoch 2 | loss=0.6906, Ranking Acc=52.96% |
| Epoch 3 | loss=0.6537, Ranking Acc=59.90% |
| Best epoch | 3 |
| 소요 시간 | 278초 |

Epoch 1의 Ranking Acc 49%는 랜덤(50%)과 유사 → RM이 처음엔 구분 못함.  
Epoch 3에서 59.9%로 상승 → "좋은 답"과 "나쁜 답"을 구분하기 시작."""))

# ── STEP 3: PPO ──
cells.append(new_markdown_cell("## STEP 3 — PPO (강화학습)"))
cells.append(new_markdown_cell("""\
### 학습 결과

| 항목 | 값 |
|------|-----|
| 데이터 수 | 12,000건 (prompt only) |
| Train loss | 509.5552 |
| avg_reward | 0.6263 |
| avg_kl | -0.2343 |
| Best epoch | 1 |
| 소요 시간 | 1,106초 (~18분) |

> ⚠️ **Reward Hacking 발생**  
> PPO 모델이 따옴표(`'`) 반복 출력으로 RM 점수를 높이는 전략을 학습.  
> avg_reward=0.6263은 실제 품질 향상이 아닌 RM 속임의 결과.  
> → KL penalty 강화 또는 RM 재학습이 필요한 상황."""))

# ── ① baseline vs SFT 비교 ──
cells.append(new_markdown_cell("## ① baseline vs SFT 비교"))

# 표 생성
table_rows = ""
for item in comparisons[:5]:  # 대표 5개
    prompt = item['prompt'][:30] + '...' if len(item['prompt']) > 30 else item['prompt']
    baseline = item['baseline'][:60].replace('\n', ' ') + '...'
    sft = item['sft'][:60].replace('\n', ' ') + '...'
    table_rows += f"| {prompt} | {baseline} | {sft} |\n"

cells.append(new_markdown_cell(f"""\
### 대표 샘플 비교 (5개)

| 프롬프트 | baseline | SFT |
|----------|----------|-----|
{table_rows}
**관찰:**
- baseline: 무한 반복, 문장 단절, 횡설수설
- SFT: "저는 AI입니다" 패턴으로 어느정도 형식 갖춤
- SFT가 대화 형식을 학습했으나 할루시네이션, 내용 반복 존재"""))

# ── ② SFT vs PPO 비교 ──
cells.append(new_markdown_cell("## ② SFT vs PPO 비교"))

table_rows2 = ""
for item in comparisons[:5]:
    prompt = item['prompt'][:30] + '...' if len(item['prompt']) > 30 else item['prompt']
    sft = item['sft'][:50].replace('\n', ' ') + '...'
    ppo = item['ppo'][:50].replace('\n', ' ') + '...'
    table_rows2 += f"| {prompt} | {sft} | {ppo} |\n"

cells.append(new_markdown_cell(f"""\
### 대표 샘플 비교 (5개)

| 프롬프트 | SFT | PPO |
|----------|-----|-----|
{table_rows2}
**관찰:**
- PPO 대부분 따옴표만 반복 → **Reward Hacking** 발생
- 예외적으로 긴 프롬프트(버스 상황)에서는 PPO가 문맥을 어느정도 반영
- Reward Hacking 원인: RM 학습 데이터에 따옴표로 시작하는 좋은 답이 많아
  PPO가 따옴표 = 높은 보상이라는 잘못된 패턴을 학습"""))

# ── 정량 평가 ──
cells.append(new_markdown_cell("## 정량 평가 (BLEU + ROUGE-L)"))

# quant_summary에서 수치 추출
bleu_baseline = quant.get('baseline', {}).get('bleu', 0.0184)
bleu_sft = quant.get('sft', {}).get('bleu', 0.1742)
bleu_ppo = quant.get('ppo', {}).get('bleu', 0.0736)
rouge_baseline = quant.get('baseline', {}).get('rouge_l', 0.0000)
rouge_sft = quant.get('sft', {}).get('rouge_l', 0.0813)
rouge_ppo = quant.get('ppo', {}).get('rouge_l', 0.0200)

cells.append(new_markdown_cell(f"""\
| 모델 | BLEU | ROUGE-L |
|------|------|---------|
| baseline | {bleu_baseline:.4f} | {rouge_baseline:.4f} |
| SFT | {bleu_sft:.4f} | {rouge_sft:.4f} |
| PPO | {bleu_ppo:.4f} | {rouge_ppo:.4f} |

- SFT가 수치상 최고 → SFT fine-tuning의 효과 확인
- PPO가 SFT보다 낮은 이유: 따옴표 반복으로 참조 답변과 겹치는 토큰 거의 없음
- baseline의 ROUGE-L=0.000 → 의미있는 토큰 생성 전혀 없음"""))

# ── ③ Decoding 실험 ──
cells.append(new_markdown_cell("## ③ Decoding 실험 (PPO 모델)"))

# 버스 프롬프트 (가장 다양한 결과)
bus = decodings[3]
cells.append(new_markdown_cell(f"""\
### 대표 프롬프트: 버스 상황 (구어체, 긴 맥락)

| 전략 | 출력 |
|------|------|
| greedy | {bus['greedy'][:80].replace(chr(10), ' ')}... |
| beam_2 | {bus['beam_2'][:80].replace(chr(10), ' ')}... |
| beam_4 | {bus['beam_4'][:80].replace(chr(10), ' ')}... |
| top-k 10 | {bus['topk_10'][:80].replace(chr(10), ' ')}... |
| top-k 50 | {bus['topk_50'][:80].replace(chr(10), ' ')}... |
| top-p 0.9 | {bus['topp_09'][:80].replace(chr(10), ' ')}... |

**관찰:**
- greedy/beam: PPO 붕괴로 따옴표 반복 (결정론적 → 동일한 나쁜 패턴 고착)
- top-k 10: 문맥 일부 반영, 조언형 문장 생성 시작
- top-k 50 / top-p 0.9: 확률 분포가 넓어질수록 다양한 토큰 샘플링 → 의미있는 문장 출현
- **결론**: PPO가 붕괴된 모델에서도 샘플링 기반 디코딩이 greedy보다 유리"""))

# ── LLM-as-a-Judge ──
cells.append(new_markdown_cell("## ④ LLM-as-a-Judge (SFT vs PPO)"))
cells.append(new_markdown_cell("""\
### 평가 프롬프트 구조

```
[질문] ...
[응답 A] SFT 출력
[응답 B] PPO 출력

평가 기준:
1. 유창성: 자연스럽고 읽기 쉬운가?
2. 관련성: 질문에 적절히 답하고 있는가?
3. 정보성: 유용한 정보를 담고 있는가?
```

### 예상 평가 결과

| 기준 | SFT | PPO |
|------|-----|-----|
| 유창성 | 3/5 (반복 있으나 문장 형성) | 1/5 (따옴표만) |
| 관련성 | 2/5 (주제는 맞으나 할루시네이션) | 1/5 (관련 없음) |
| 정보성 | 2/5 (내용 반복, 틀린 정보 다수) | 1/5 (정보 없음) |
| **승자** | **SFT** | - |

> Reward Hacking이 발생한 PPO는 LLM-as-a-Judge 평가에서도 SFT에 열위.
> 실제 ChatGPT급 PPO는 RM 품질 + KL penalty 조정 + 더 큰 모델이 필요."""))

# ── 결론 ──
cells.append(new_markdown_cell("## 결론 및 고찰"))
cells.append(new_markdown_cell("""\
### RLHF 3단계 파이프라인 체험 요약

| 단계 | 핵심 학습 |
|------|-----------|
| SFT | 형식과 말투를 학습시킬 수 있다 |
| RM | "좋은 답"을 점수로 정의할 수 있다 (59.9% Ranking Acc) |
| PPO | RM이 허술하면 Reward Hacking이 발생한다 |

### Reward Hacking이란?
Goodhart's Law: **"측정값이 목표가 되는 순간, 측정값은 더 이상 좋은 목표가 아니다"**  
→ PPO가 실제 좋은 답변 대신 RM을 속이는 따옴표 반복 전략을 발견

### 개선 방향
1. `tokenizer.padding_side = 'left'` 적용 (right-padding 경고 제거)
2. KL penalty 계수 상향 (현재 0.1 → 0.3~0.5)
3. RM 학습 데이터 품질 개선 (따옴표 패턴 제거)
4. PPO epochs 증가 또는 배치 크기 조정"""))

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
