"""
실험 자동 추적 시스템
- 매 학습마다 하이퍼파라미터, 데이터 설정, 학습 결과를 자동 기록
- CSV / JSON 이중 저장
- 학습 완료 후 이전 실험 대비 성능 변화 터미널 출력
"""

import os
import csv
import json
import sys
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없는 환경(서버) 대응
    import matplotlib.pyplot as plt
    import numpy as np
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# val_loss 가 train_loss 보다 이 비율 이상 높으면 오버피팅으로 판단
OVERFITTING_THRESHOLD = 0.15

LOG_CSV  = os.path.join(config.MODEL_PATH, 'experiments_log.csv')
LOG_JSON = os.path.join(config.MODEL_PATH, 'experiments_log.json')


class ExperimentTracker:
    """실험 자동 추적 클래스"""

    def __init__(self):
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        self.csv_path  = LOG_CSV
        self.json_path = LOG_JSON
        self.experiment_id = self._next_id()
        self.start_time = datetime.now()
        print(f"[ExperimentTracker] 실험 #{self.experiment_id} 시작 ({self.start_time.strftime('%Y-%m-%d %H:%M:%S')})")

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _load_all(self):
        """JSON에서 전체 실험 기록 로드"""
        if not os.path.exists(self.json_path):
            return []
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def _next_id(self):
        records = self._load_all()
        return max((r['experiment_id'] for r in records), default=0) + 1

    def _detect_overfitting(self, train_losses, val_losses):
        """
        마지막 에폭의 (val - train) / train 이 임계치 초과 시 오버피팅으로 판단.
        val < train 이면 오버피팅 아님.
        """
        if not train_losses or not val_losses:
            return False
        t, v = train_losses[-1], val_losses[-1]
        if t <= 0:
            return False
        return (v - t) / t > OVERFITTING_THRESHOLD

    # ------------------------------------------------------------------
    # 핵심 메서드
    # ------------------------------------------------------------------

    def build_record(self, train_losses, val_losses, best_epoch,
                     tokenizer_name='unknown',
                     original_data_size=None, augmented_data_size=None):
        """실험 기록 dict 생성 (저장 전 검사용으로도 호출 가능)"""

        final_train = round(train_losses[-1], 6) if train_losses else None
        final_val   = round(val_losses[-1],   6) if val_losses   else None
        best_val    = round(min(val_losses),   6) if val_losses   else None

        data_synthesized = (
            augmented_data_size is not None
            and original_data_size is not None
            and augmented_data_size > original_data_size
        )

        return {
            # 식별 정보
            'experiment_id':   self.experiment_id,
            'timestamp':       self.start_time.isoformat(),
            'duration_sec':    round((datetime.now() - self.start_time).total_seconds(), 1),

            # 하이퍼파라미터
            'num_layers':         config.TRANSFORMER_NUM_LAYERS,
            'd_model':            config.TRANSFORMER_D_MODEL,
            'nhead':              config.TRANSFORMER_NHEAD,
            'dim_feedforward':    config.TRANSFORMER_DIM_FEEDFORWARD,
            'feedforward_ratio':  round(config.TRANSFORMER_DIM_FEEDFORWARD / config.TRANSFORMER_D_MODEL, 2),
            'dropout':            config.DROPOUT_RATE,
            'learning_rate':      config.LEARNING_RATE,
            'batch_size':         config.BATCH_SIZE,
            'epochs':             config.EPOCHS,
            'max_seq_length':     config.MAX_SEQ_LENGTH,
            'vocab_size_cfg':     config.VOCAB_SIZE,

            # 데이터 설정
            'use_augmentation':    getattr(config, 'USE_AUGMENTATION', True),
            'data_synthesized':    data_synthesized,
            'original_data_size':  original_data_size,
            'augmented_data_size': augmented_data_size,
            'tokenizer':           tokenizer_name,

            # 학습 결과
            'final_train_loss': final_train,
            'final_val_loss':   final_val,
            'best_val_loss':    best_val,
            'best_epoch':       best_epoch,
            'overfitting':      self._detect_overfitting(train_losses, val_losses),
        }

    def save(self, train_losses, val_losses, best_epoch,
             tokenizer_name='unknown',
             original_data_size=None, augmented_data_size=None):
        """
        실험 기록을 CSV + JSON 에 저장하고 터미널에 비교 결과를 출력한다.

        Args:
            train_losses (list[float]): 에폭별 train loss
            val_losses   (list[float]): 에폭별 val loss
            best_epoch   (int): best val_loss 를 기록한 에폭 번호 (1-based)
            tokenizer_name (str): 'Mecab' | 'Okt' | 'whitespace' | 'unknown'
            original_data_size  (int): 전처리 후 원본 데이터 행 수
            augmented_data_size (int): 증강 후 데이터 행 수
        """
        record    = self.build_record(train_losses, val_losses, best_epoch,
                                      tokenizer_name, original_data_size, augmented_data_size)
        all_prev  = self._load_all()
        all_new   = all_prev + [record]

        # JSON 저장
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new, f, ensure_ascii=False, indent=2)

        # CSV 저장 (헤더는 첫 실험에만)
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        print(f"\n[ExperimentTracker] 실험 #{self.experiment_id} 저장 완료")
        print(f"  CSV : {self.csv_path}")
        print(f"  JSON: {self.json_path}")

        self._print_summary(record, all_prev)
        self._generate_plots(record, all_new)
        return record

    # ------------------------------------------------------------------
    # 시각화
    # ------------------------------------------------------------------

    def _generate_plots(self, record, all_records):
        """학습 완료 후 3종 그래프를 models/ 에 자동 저장"""
        if not _PLOT_AVAILABLE:
            print("[ExperimentTracker] matplotlib 없음 — 그래프 생성 건너뜀")
            return

        plot_dir = config.MODEL_PATH
        os.makedirs(plot_dir, exist_ok=True)

        # ── 1. training_loss.png (현재 실험 Train/Val Loss 곡선) ──────
        self._plot_training_loss(plot_dir)

        # ── 2. experiments_comparison.png (전체 실험 Val Loss 막대) ──
        self._plot_experiments_comparison(all_records, plot_dir)

        # ── 3. heatmap.png (하이퍼파라미터 vs Val Loss) ──────────────
        self._plot_heatmap(all_records, plot_dir)

        print(f"[ExperimentTracker] 그래프 저장 완료 → {plot_dir}/")

    def _plot_training_loss(self, plot_dir):
        """현재 실험의 Train/Val Loss 곡선 — experiments_log.csv 마지막 행 참조"""
        if not os.path.exists(self.json_path):
            return
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
        except Exception:
            return

        # training_results.json 에서 에폭별 loss 읽기
        results_path = os.path.join(config.MODEL_PATH, 'training_results.json')
        if not os.path.exists(results_path):
            return
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            return

        train_losses = results.get('train_losses', [])
        val_losses   = results.get('val_losses', [])
        if not train_losses:
            return

        epochs = range(1, len(train_losses) + 1)
        exp_id = all_records[-1]['experiment_id'] if all_records else '?'

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
        ax.plot(epochs, val_losses,   label='Val Loss',   marker='s', markersize=3)
        best_epoch = results.get('best_val_loss')
        if best_epoch is not None:
            ax.axhline(y=best_epoch, color='gray', linestyle='--', linewidth=0.8, label=f'Best Val {best_epoch:.4f}')
        ax.set_title(f'Experiment #{exp_id} — Train / Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'training_loss.png'), dpi=150)
        plt.close(fig)

    def _plot_experiments_comparison(self, all_records, plot_dir):
        """전체 실험 Best Val Loss 막대그래프"""
        if not all_records:
            return

        ids   = [r['experiment_id'] for r in all_records]
        vals  = [r.get('best_val_loss') or 0 for r in all_records]
        colors = ['steelblue'] * (len(ids) - 1) + ['tomato']  # 최신 실험 강조

        fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.8 + 2), 5))
        bars = ax.bar([str(i) for i in ids], vals, color=colors)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
        ax.set_title('All Experiments — Best Val Loss')
        ax.set_xlabel('Experiment #')
        ax.set_ylabel('Best Val Loss')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'experiments_comparison.png'), dpi=150)
        plt.close(fig)

    def _plot_heatmap(self, all_records, plot_dir):
        """하이퍼파라미터 vs Val Loss 히트맵"""
        if not all_records:
            return

        cols = ['dropout', 'learning_rate', 'epochs', 'best_val_loss']
        labels = ['Dropout', 'LR', 'Epochs', 'Best Val Loss']

        data = []
        row_labels = []
        for r in all_records:
            row = [r.get(c) for c in cols]
            if any(v is None for v in row):
                continue
            data.append(row)
            row_labels.append(f"#{r['experiment_id']}")

        if not data:
            return

        arr = np.array(data, dtype=float)
        # 열별 min-max 정규화 (시각적 대비를 위해)
        arr_norm = np.zeros_like(arr)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            mn, mx = col.min(), col.max()
            arr_norm[:, j] = (col - mn) / (mx - mn) if mx != mn else np.zeros_like(col)

        fig, ax = plt.subplots(figsize=(7, max(3, len(row_labels) * 0.5 + 1.5)))
        im = ax.imshow(arr_norm, aspect='auto', cmap='RdYlGn_r')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

        # 셀에 실제 값 표시
        for i in range(len(row_labels)):
            for j in range(len(labels)):
                val = arr[i, j]
                text = f'{val:.4f}' if j == len(labels) - 1 else (
                    f'{val:.4f}' if j == 1 else f'{val:.2f}' if j == 0 else f'{int(val)}'
                )
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                        color='black' if 0.2 < arr_norm[i, j] < 0.8 else 'white')

        ax.set_title('Hyperparameter Heatmap (normalized)')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'heatmap.png'), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 터미널 출력
    # ------------------------------------------------------------------

    def _print_summary(self, current, previous_records):
        SEP = "=" * 62

        print(f"\n{SEP}")
        print(f"  실험 #{current['experiment_id']}  결과 요약")
        print(SEP)

        print(f"  {'날짜/시간':<20} {current['timestamp'][:19]}")
        print(f"  {'소요 시간':<20} {current['duration_sec']} 초")
        print()
        print(f"  [하이퍼파라미터]")
        print(f"  {'레이어 수':<20} {current['num_layers']}")
        print(f"  {'d_model':<20} {current['d_model']}")
        print(f"  {'피드포워드 차원':<20} {current['dim_feedforward']}  (x{current['feedforward_ratio']})")
        print(f"  {'nhead':<20} {current['nhead']}")
        print(f"  {'드롭아웃':<20} {current['dropout']}")
        print(f"  {'러닝레이트':<20} {current['learning_rate']}")
        print(f"  {'배치 사이즈':<20} {current['batch_size']}")
        print(f"  {'에폭':<20} {current['epochs']}")
        print(f"  {'MAX_SEQ_LENGTH':<20} {current['max_seq_length']}")
        print()
        print(f"  [데이터 설정]")
        print(f"  {'증강 여부':<20} {current['use_augmentation']}")
        print(f"  {'합성 데이터':<20} {current['data_synthesized']}")
        print(f"  {'토크나이저':<20} {current['tokenizer']}")
        print(f"  {'원본 데이터 크기':<20} {current['original_data_size']}")
        print(f"  {'증강 데이터 크기':<20} {current['augmented_data_size']}")
        print()
        print(f"  [학습 결과]")
        print(f"  {'Best Val Loss':<20} {current['best_val_loss']}")
        print(f"  {'Final Train Loss':<20} {current['final_train_loss']}")
        print(f"  {'Final Val Loss':<20} {current['final_val_loss']}")
        print(f"  {'Best Epoch':<20} {current['best_epoch']} / {current['epochs']}")
        overfit_str = "Yes  <-- 오버피팅 의심" if current['overfitting'] else "No"
        print(f"  {'오버피팅':<20} {overfit_str}")

        if not previous_records:
            print(f"\n  (첫 번째 실험 - 이전 비교 없음)")
            print(SEP)
            return

        prev = previous_records[-1]
        print(f"\n  --- 이전 실험 #{prev['experiment_id']} 대비 변화 ---")

        def delta(key, higher_is_better=False):
            c, p = current.get(key), prev.get(key)
            if c is None or p is None:
                return 'N/A'
            if c == p:
                return f'{c}  (변화없음)'
            diff = c - p
            improved = (diff < 0) if not higher_is_better else (diff > 0)
            sign = '+' if diff > 0 else ''
            tag  = '(개선)' if improved else '(악화)'
            return f'{c}  [{sign}{diff:+.6f} {tag}]'

        print(f"  {'Best Val Loss':<20} {delta('best_val_loss')}")
        print(f"  {'Final Train Loss':<20} {delta('final_train_loss')}")
        print(f"  {'Final Val Loss':<20} {delta('final_val_loss')}")

        # 하이퍼파라미터 변경 사항
        hp_keys = [
            'num_layers', 'd_model', 'dim_feedforward', 'feedforward_ratio',
            'dropout', 'learning_rate', 'batch_size', 'epochs', 'max_seq_length',
        ]
        changed = [(k, prev.get(k), current.get(k))
                   for k in hp_keys if prev.get(k) != current.get(k)]
        if changed:
            print(f"\n  [하이퍼파라미터 변경]")
            for k, old, new in changed:
                print(f"    {k:<22} {old}  -->  {new}")
        else:
            print(f"\n  [하이퍼파라미터] 이전 실험과 동일")

        print(SEP)
