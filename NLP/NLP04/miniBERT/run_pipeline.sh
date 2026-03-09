#!/bin/bash
# ============================================================
# mini BERT Pretrain 전체 파이프라인
#
# 사용법:
#   bash run_pipeline.sh                        # 기본값 사용
#   bash run_pipeline.sh --max_samples 100000   # 샘플 수 지정
# ============================================================

set -e   # 오류 발생 시 즉시 중단

# ── 상단 변수 (필요 시 수정) ─────────────────────────────────
MAX_SAMPLES=50000   # 기본 샘플 수 (0 이면 전체 코퍼스 사용)

# ── 인자 파싱 ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_pipeline.sh [--max_samples N]"
            exit 1
            ;;
    esac
done

# ── 경로 설정 ────────────────────────────────────────────────
cd /workspace/NLP04

LOG_DIR="/workspace/NLP04/logs"
mkdir -p "$LOG_DIR"

# ── 헬퍼 함수 ────────────────────────────────────────────────
ts() { date +'%Y-%m-%d %H:%M:%S'; }

banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

log_pipeline() {
    echo "[$(ts)] $1" >> "$LOG_DIR/pipeline.log"
}

# ── 파이프라인 시작 ──────────────────────────────────────────
banner "mini BERT Pretrain Pipeline 시작"
echo "  시작 시간  : $(ts)"
echo "  MAX_SAMPLES: ${MAX_SAMPLES} (0=전체)"
echo "  로그 디렉토리: $LOG_DIR"
echo ""

> "$LOG_DIR/pipeline.log"   # 파이프라인 로그 초기화
log_pipeline "파이프라인 시작 (MAX_SAMPLES=${MAX_SAMPLES})"

# ============================================================
# Step 1: Build Tokenizer
# ============================================================
banner "Step 1: SentencePiece 토크나이저 학습"
echo "  시작: $(ts)"
log_pipeline "Step 1 시작"

python scripts/01_build_tokenizer.py 2>&1 | tee "$LOG_DIR/01_tokenizer.log"

echo "  완료: $(ts)"
log_pipeline "Step 1 완료"

# special token ID 검증 결과 확인
echo ""
echo "[검증] special token IDs:"
grep -E '\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\]' "$LOG_DIR/01_tokenizer.log" || true

# ============================================================
# Step 2: Preprocess Data
# ============================================================
banner "Step 2: 데이터 전처리 (MLM + NSP)"
echo "  시작: $(ts)"
log_pipeline "Step 2 시작"

if [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
    python scripts/02_preprocess_data.py --max_samples "$MAX_SAMPLES" 2>&1 | tee "$LOG_DIR/02_preprocess.log"
else
    python scripts/02_preprocess_data.py 2>&1 | tee "$LOG_DIR/02_preprocess.log"
fi

echo "  완료: $(ts)"
log_pipeline "Step 2 완료"

# ============================================================
# Step 3: Pretrain
# ============================================================
banner "Step 3: mini BERT 사전학습"
echo "  시작: $(ts)"
log_pipeline "Step 3 시작"

python scripts/03_pretrain.py 2>&1 | tee "$LOG_DIR/03_pretrain.log"

echo "  완료: $(ts)"
log_pipeline "Step 3 완료"

# loss 그래프 저장 확인
echo ""
echo "[확인] 생성된 모델 파일:"
PLOT_PATH="/workspace/NLP04/models/pretrain_loss.png"
if [ -f "$PLOT_PATH" ]; then
    SIZE=$(du -h "$PLOT_PATH" | cut -f1)
    echo "  pretrain_loss.png  ($SIZE)  OK"
else
    echo "  pretrain_loss.png  NOT FOUND"
fi

BEST_CKPT="/workspace/NLP04/models/bert_best.pt"
if [ -f "$BEST_CKPT" ]; then
    SIZE=$(du -h "$BEST_CKPT" | cut -f1)
    echo "  bert_best.pt       ($SIZE)  OK"
fi

FINAL_CKPT="/workspace/NLP04/models/bert_final.pt"
if [ -f "$FINAL_CKPT" ]; then
    SIZE=$(du -h "$FINAL_CKPT" | cut -f1)
    echo "  bert_final.pt      ($SIZE)  OK"
fi

# ============================================================
# 완료 요약
# ============================================================
banner "파이프라인 완료!"
log_pipeline "파이프라인 완료"

echo "  종료 시간 : $(ts)"
echo ""
echo "  생성된 파일:"
echo "    data/processed/spm.model        - SentencePiece 토크나이저"
echo "    data/processed/spm.vocab        - 어휘 목록"
echo "    data/processed/input_ids.dat    - MLM 입력 (memmap)"
echo "    data/processed/mlm_labels.dat   - MLM 레이블 (memmap)"
echo "    data/processed/nsp_label.dat    - NSP 레이블 (memmap)"
echo "    models/bert_best.pt             - Best 체크포인트"
echo "    models/bert_final.pt            - 최종 체크포인트"
echo "    models/pretrain_loss.png        - Loss 그래프"
echo "    models/experiments_log.csv      - 실험 로그 CSV"
echo "    models/experiments_log.json     - 실험 로그 JSON"
echo ""
echo "  로그 파일:"
echo "    logs/01_tokenizer.log"
echo "    logs/02_preprocess.log"
echo "    logs/03_pretrain.log"
echo "    logs/pipeline.log"
echo ""
echo "  파이프라인 로그:"
cat "$LOG_DIR/pipeline.log"
echo "============================================================"
