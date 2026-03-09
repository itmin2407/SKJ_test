"""
02_preprocess_data.py
MLM + NSP 전처리 후 np.memmap으로 저장

필드: input_ids, segment_ids, attention_mask, mlm_labels, nsp_label
"""

import os
os.chdir('/workspace/NLP04')

import json
import random
import numpy as np
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR   = '/workspace/NLP04'
RAW_DIR    = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR   = os.path.join(BASE_DIR, 'data', 'processed')
SPM_MODEL  = os.path.join(PROC_DIR, 'spm.model')
CORPUS_TXT = os.path.join(RAW_DIR, 'corpus.txt')

os.makedirs(PROC_DIR, exist_ok=True)

with open(os.path.join(BASE_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    CFG = json.load(f)

MAX_SEQ_LEN = CFG['max_seq_len']
VOCAB_SIZE  = CFG['vocab_size']

# special token IDs (sentencepiece 기준)
PAD_ID  = 0
UNK_ID  = 1
CLS_ID  = 2
SEP_ID  = 3
MASK_ID = 4

# MLM 비율
MLM_PROB  = 0.15
MASK_PROB = 0.80   # 선택된 토큰 중 80% → [MASK]
RAND_PROB = 0.10   # 10% → 랜덤 토큰
# 나머지 10% → 원본 유지

random.seed(42)
np.random.seed(42)


# ── SentencePiece 로드 ────────────────────────────────────────
def load_tokenizer():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL)
    return sp


# ── 코퍼스 로드 (문장 리스트) ─────────────────────────────────
def load_sentences():
    print("[load] reading corpus.txt ...")
    sentences = []
    with open(CORPUS_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 5:
                sentences.append(line)
    print(f"[load] {len(sentences):,} sentences loaded")
    return sentences


# ── MLM 마스킹 ────────────────────────────────────────────────
def apply_mlm(token_ids: list, vocab_size: int):
    """
    Returns:
        masked_ids : 마스킹 적용된 입력
        mlm_labels : 마스킹 위치만 실제 ID, 나머지는 -100
    """
    masked_ids  = list(token_ids)
    mlm_labels  = [-100] * len(token_ids)

    # [CLS], [SEP]는 마스킹 대상에서 제외
    candidate_idx = [
        i for i, tid in enumerate(token_ids)
        if tid not in (CLS_ID, SEP_ID, PAD_ID)
    ]

    n_mask = max(1, int(len(candidate_idx) * MLM_PROB))
    selected = random.sample(candidate_idx, min(n_mask, len(candidate_idx)))

    for idx in selected:
        orig_id = token_ids[idx]
        mlm_labels[idx] = orig_id   # 실제 레이블 기록

        r = random.random()
        if r < MASK_PROB:
            masked_ids[idx] = MASK_ID
        elif r < MASK_PROB + RAND_PROB:
            masked_ids[idx] = random.randint(5, vocab_size - 1)
        # else: 원본 유지

    return masked_ids, mlm_labels


# ── 샘플 생성 ─────────────────────────────────────────────────
def build_sample(sp, sent_a_ids: list, sent_b_ids: list, nsp_label: int, max_len: int):
    """
    [CLS] A [SEP] B [SEP] 형식으로 조합 후 패딩/트런케이션.

    Returns dict with keys:
        input_ids, segment_ids, attention_mask, mlm_labels, nsp_label
    """
    # 최대 길이 맞춰 A, B 트런케이션 (균등하게)
    max_ab = max_len - 3  # [CLS] + [SEP] + [SEP]
    while len(sent_a_ids) + len(sent_b_ids) > max_ab:
        if len(sent_a_ids) > len(sent_b_ids):
            sent_a_ids.pop()
        else:
            sent_b_ids.pop()

    # 시퀀스 조합
    token_ids = [CLS_ID] + sent_a_ids + [SEP_ID] + sent_b_ids + [SEP_ID]

    # segment ids: A=0, B=1
    seg_len_a = 1 + len(sent_a_ids) + 1   # [CLS] + A + [SEP]
    seg_len_b = len(sent_b_ids) + 1        # B + [SEP]
    segment_ids = [0] * seg_len_a + [1] * seg_len_b

    seq_len = len(token_ids)
    attention_mask = [1] * seq_len

    # 패딩
    pad_len = max_len - seq_len
    token_ids      += [PAD_ID] * pad_len
    segment_ids    += [0]      * pad_len
    attention_mask += [0]      * pad_len

    # MLM 적용
    masked_ids, mlm_labels = apply_mlm(token_ids, VOCAB_SIZE)

    return {
        'input_ids':      masked_ids,
        'segment_ids':    segment_ids,
        'attention_mask': attention_mask,
        'mlm_labels':     mlm_labels,
        'nsp_label':      nsp_label,
    }


# ── 데이터셋 생성 ─────────────────────────────────────────────
def build_dataset(sp, sentences, max_samples=None):
    print("[build] generating NSP + MLM samples ...")
    samples = []
    n = len(sentences)

    # 모든 문장을 토크나이즈
    print("[tokenize] encoding all sentences ...")
    enc_sentences = [sp.EncodeAsIds(s) for s in tqdm(sentences, desc='tokenize')]

    indices = list(range(n - 1))
    random.shuffle(indices)
    if max_samples:
        indices = indices[:max_samples * 2]   # IsNext + NotNext 합산 고려

    for i in tqdm(indices, desc='build samples'):
        # IsNext (50%)
        a_ids = list(enc_sentences[i])
        b_ids = list(enc_sentences[i + 1])
        samples.append(build_sample(sp, a_ids, b_ids, nsp_label=1, max_len=MAX_SEQ_LEN))

        # NotNext (50%)
        j = random.randint(0, n - 1)
        while j == i or j == i + 1:
            j = random.randint(0, n - 1)
        a_ids2 = list(enc_sentences[i])
        b_ids2 = list(enc_sentences[j])
        samples.append(build_sample(sp, a_ids2, b_ids2, nsp_label=0, max_len=MAX_SEQ_LEN))

        if max_samples and len(samples) >= max_samples:
            break

    random.shuffle(samples)
    print(f"[build] total samples: {len(samples):,}")
    return samples


# ── np.memmap 저장 ────────────────────────────────────────────
def save_memmap(samples):
    n = len(samples)
    L = MAX_SEQ_LEN

    fields = {
        'input_ids':      ('int32',   (n, L)),
        'segment_ids':    ('int32',   (n, L)),
        'attention_mask': ('int32',   (n, L)),
        'mlm_labels':     ('int32',   (n, L)),
        'nsp_label':      ('int32',   (n,)),
    }

    print("[save] writing memmap files ...")
    memmaps = {}
    for name, (dtype, shape) in fields.items():
        path = os.path.join(PROC_DIR, f'{name}.dat')
        memmaps[name] = np.memmap(path, dtype=dtype, mode='w+', shape=shape)

    for i, s in enumerate(tqdm(samples, desc='write memmap')):
        memmaps['input_ids'][i]      = s['input_ids']
        memmaps['segment_ids'][i]    = s['segment_ids']
        memmaps['attention_mask'][i] = s['attention_mask']
        memmaps['mlm_labels'][i]     = s['mlm_labels']
        memmaps['nsp_label'][i]      = s['nsp_label']

    for mm in memmaps.values():
        mm.flush()

    # 메타데이터 저장
    meta = {'n_samples': n, 'max_seq_len': L, 'vocab_size': VOCAB_SIZE}
    with open(os.path.join(PROC_DIR, 'dataset_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[save] {n:,} samples saved to {PROC_DIR}")
    for name, (dtype, shape) in fields.items():
        mb = np.prod(shape) * np.dtype(dtype).itemsize / 1e6
        print(f"  {name}.dat  shape={shape}  {mb:.1f} MB")


# ── main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None,
                        help='샘플 수 제한 (미지정시 전체 코퍼스 사용)')
    args = parser.parse_args()

    print("=" * 60)
    print(" Step 2: Preprocess Data (MLM + NSP)")
    print("=" * 60)

    sp = load_tokenizer()
    sentences = load_sentences()
    samples = build_dataset(sp, sentences, max_samples=args.max_samples)
    save_memmap(samples)

    print("\n[all done] 02_preprocess_data.py complete")
