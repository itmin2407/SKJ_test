"""
01_build_tokenizer.py
SentencePiece BPE 모델 학습 (vocab_size=8000)
special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
"""

import os
os.chdir('/workspace/NLP04')

import sys
import json
import subprocess
import bz2
import re
import xml.etree.ElementTree as ET

# ── 경로 설정 ────────────────────────────────────────────────
BASE_DIR    = '/workspace/NLP04'
RAW_DIR     = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR    = os.path.join(BASE_DIR, 'data', 'processed')
DUMP_URL    = 'https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2'
DUMP_PATH   = os.path.join(RAW_DIR, 'kowiki-latest-pages-articles.xml.bz2')
CORPUS_TXT  = os.path.join(RAW_DIR, 'corpus.txt')
SPM_PREFIX  = os.path.join(PROC_DIR, 'spm')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

with open(os.path.join(BASE_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    CFG = json.load(f)

VOCAB_SIZE = CFG['vocab_size']


# ── 나무위키 dump 다운로드 ────────────────────────────────────
def download_dump():
    if os.path.exists(DUMP_PATH):
        print(f"[skip] dump already exists: {DUMP_PATH}")
        return
    print(f"[download] {DUMP_URL}")
    try:
        subprocess.run(
            ['wget', '-c', '-O', DUMP_PATH, DUMP_URL],
            check=True
        )
        print("[done] download complete")
    except FileNotFoundError:
        # wget 없을 경우 Python urllib 사용
        import urllib.request
        print("[info] wget not found, using urllib...")
        urllib.request.urlretrieve(DUMP_URL, DUMP_PATH)
        print("[done] download complete")


# ── XML dump → 순수 텍스트 추출 ──────────────────────────────
def extract_text_from_dump():
    if os.path.exists(CORPUS_TXT):
        size_mb = os.path.getsize(CORPUS_TXT) / 1e6
        print(f"[skip] corpus.txt already exists ({size_mb:.1f} MB)")
        return

    print("[extract] reading bz2 dump and extracting text...")
    ns = '{http://www.mediawiki.org/xml/DTD/MediaWiki}'
    line_count = 0

    # wikiextractor 사용 시도
    try:
        result = subprocess.run(
            ['python', '-m', 'wikiextractor.WikiExtractor',
             '--no-templates', '-o', '-', DUMP_PATH],
            capture_output=True, text=True, timeout=3600
        )
        with open(CORPUS_TXT, 'w', encoding='utf-8') as fw:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and not line.startswith('<'):
                    fw.write(line + '\n')
                    line_count += 1
        print(f"[done] extracted {line_count} lines via wikiextractor")
        return
    except Exception:
        pass

    # fallback: 직접 XML 파싱
    print("[info] wikiextractor not found, using direct XML parsing (slower)...")

    wiki_re = re.compile(
        r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]'   # [[링크|텍스트]] → 텍스트
        r'|\{\{[^}]*\}\}'                   # {{템플릿}} 제거
        r'|<[^>]+>',                        # HTML 태그 제거
        re.DOTALL
    )

    def clean_wiki(text):
        text = wiki_re.sub(lambda m: m.group(1) or '', text)
        text = re.sub(r'={2,}[^=]+=={2,}', '', text)  # 섹션 헤더 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    with bz2.open(DUMP_PATH, 'rt', encoding='utf-8') as fz, \
         open(CORPUS_TXT, 'w', encoding='utf-8') as fw:
        in_text = False
        buf = []
        for line in fz:
            if '<text' in line:
                in_text = True
                buf = [line]
            elif '</text>' in line and in_text:
                buf.append(line)
                raw = ''.join(buf)
                # text 태그 내용 추출
                m = re.search(r'<text[^>]*>(.*?)</text>', raw, re.DOTALL)
                if m:
                    content = clean_wiki(m.group(1))
                    for sent in re.split(r'[.!?。]\s+', content):
                        sent = sent.strip()
                        if len(sent) > 10:
                            fw.write(sent + '\n')
                            line_count += 1
                in_text = False
                buf = []
            elif in_text:
                buf.append(line)

    print(f"[done] extracted {line_count} sentences")


# ── SentencePiece 학습 ────────────────────────────────────────
def train_sentencepiece():
    model_path = SPM_PREFIX + '.model'
    if os.path.exists(model_path):
        print(f"[skip] spm.model already exists: {model_path}")
        return

    try:
        import sentencepiece as spm
    except ImportError:
        print("[install] sentencepiece...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'sentencepiece',
             '--break-system-packages'],
            check=True
        )
        import sentencepiece as spm

    print(f"[train] SentencePiece BPE, vocab_size={VOCAB_SIZE} ...")

    spm.SentencePieceTrainer.train(
        input=CORPUS_TXT,
        model_prefix=SPM_PREFIX,
        model_type='bpe',
        vocab_size=VOCAB_SIZE,
        character_coverage=0.9995,
        pad_id=0,          # [PAD]
        unk_id=1,          # [UNK]
        bos_id=2,          # [CLS]
        eos_id=3,          # [SEP]
        user_defined_symbols=['[MASK]'],   # [MASK]=4
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[CLS]',
        eos_piece='[SEP]',
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 4,
    )
    print(f"[done] saved: {SPM_PREFIX}.model, {SPM_PREFIX}.vocab")


# ── 검증 ─────────────────────────────────────────────────────
def verify_tokenizer():
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_PREFIX + '.model')

    print("\n=== Tokenizer Verification ===")
    print(f"vocab_size : {sp.GetPieceSize()}")
    for name, idx in [('[PAD]', 0), ('[UNK]', 1), ('[CLS]', 2),
                      ('[SEP]', 3), ('[MASK]', 4)]:
        actual_id = sp.PieceToId(name)
        status = 'OK' if actual_id == idx else f'MISMATCH (got {actual_id})'
        print(f"  {name:8s} id={idx} → {status}")

    sample = "안녕하세요. 자연어 처리 미니 BERT 사전학습 프로젝트입니다."
    tokens = sp.EncodeAsPieces(sample)
    ids    = sp.EncodeAsIds(sample)
    print(f"\nSample: {sample}")
    print(f"Tokens: {tokens}")
    print(f"IDs   : {ids}")


# ── main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print(" Step 1: Build SentencePiece Tokenizer")
    print("=" * 60)

    download_dump()
    extract_text_from_dump()
    train_sentencepiece()
    verify_tokenizer()

    print("\n[all done] 01_build_tokenizer.py complete")
