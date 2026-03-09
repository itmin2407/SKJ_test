"""
experiment_tracker.py
mini BERT 사전학습 실험 자동 추적 시스템

기록 항목:
- 실험 ID (timestamp 기반)
- config.json 스냅샷 (실험 시점 하이퍼파라미터 전체)
- total parameters 수
- 각 epoch별 mlm_loss, nsp_loss, total_loss, mlm_acc, nsp_acc
- 총 학습 시간
- best epoch (최저 total_loss 기준)

저장 경로:
- /workspace/NLP04/models/experiments_log.csv
- /workspace/NLP04/models/experiments_log.json
"""

import os
import csv
import json
import math
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

BASE_DIR   = '/workspace/NLP04'
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
LOG_CSV    = os.path.join(MODEL_DIR, 'experiments_log.csv')
LOG_JSON   = os.path.join(MODEL_DIR, 'experiments_log.json')


class ExperimentTracker:
    """mini BERT 사전학습 실험 추적 클래스"""

    def __init__(self, cfg: dict, total_params: int):
        """
        Args:
            cfg          : config.json 전체 dict (하이퍼파라미터 스냅샷)
            total_params : 모델 전체 파라미터 수
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.cfg          = cfg
        self.total_params = total_params
        self.start_time   = datetime.now()
        self.epoch_logs   = []   # 매 epoch 결과 누적

        self.experiment_id = self._next_id()
        print(
            f"\n[ExperimentTracker] 실험 #{self.experiment_id} 시작 "
            f"({self.start_time.strftime('%Y-%m-%d %H:%M:%S')})"
        )

    # ── 내부 유틸 ────────────────────────────────────────────────

    def _load_all(self) -> list:
        if not os.path.exists(LOG_JSON):
            return []
        try:
            with open(LOG_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def _next_id(self) -> int:
        records = self._load_all()
        return max((r['experiment_id'] for r in records), default=0) + 1

    # ── 공개 API ─────────────────────────────────────────────────

    def log_epoch(self, epoch: int, stats: dict):
        """
        매 epoch 호출.

        Args:
            epoch : 현재 epoch 번호 (1-based)
            stats : train_epoch() 반환 dict
                    {total_loss, mlm_loss, nsp_loss, mlm_acc, nsp_acc}
        """
        entry = {'epoch': epoch, **stats}
        self.epoch_logs.append(entry)

    def save(self, best_epoch: int):
        """
        학습 완료 후 호출. CSV + JSON 에 기록하고 터미널 요약 출력.

        Args:
            best_epoch : 최저 total_loss를 기록한 epoch 번호
        """
        duration = round((datetime.now() - self.start_time).total_seconds(), 1)

        record = self._build_record(best_epoch, duration)
        all_prev = self._load_all()
        all_new  = all_prev + [record]

        # JSON 저장
        with open(LOG_JSON, 'w', encoding='utf-8') as f:
            json.dump(all_new, f, ensure_ascii=False, indent=2)

        # CSV 저장
        write_header = not os.path.exists(LOG_CSV)
        flat = self._flatten(record)
        with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat)

        print(f"\n[ExperimentTracker] 실험 #{self.experiment_id} 저장 완료")
        print(f"  CSV : {LOG_CSV}")
        print(f"  JSON: {LOG_JSON}")

        self._print_summary(record, all_prev)
        self._generate_plots(all_new)
        return record

    # ── 기록 생성 ────────────────────────────────────────────────

    def _build_record(self, best_epoch: int, duration_sec: float) -> dict:
        best_stats  = min(self.epoch_logs, key=lambda x: x['total_loss'])
        final_stats = self.epoch_logs[-1] if self.epoch_logs else {}

        return {
            # 식별 정보
            'experiment_id': self.experiment_id,
            'timestamp':     self.start_time.isoformat(),
            'duration_sec':  duration_sec,

            # 모델 규모
            'total_params':  self.total_params,

            # config 스냅샷 (하이퍼파라미터 전체)
            'config': self.cfg,

            # 학습 요약
            'best_epoch':       best_epoch,
            'best_total_loss':  round(best_stats.get('total_loss', 0), 6),
            'best_mlm_loss':    round(best_stats.get('mlm_loss',   0), 6),
            'best_nsp_loss':    round(best_stats.get('nsp_loss',   0), 6),
            'best_mlm_acc':     round(best_stats.get('mlm_acc',    0), 6),
            'best_nsp_acc':     round(best_stats.get('nsp_acc',    0), 6),

            'final_total_loss': round(final_stats.get('total_loss', 0), 6),
            'final_mlm_loss':   round(final_stats.get('mlm_loss',   0), 6),
            'final_nsp_loss':   round(final_stats.get('nsp_loss',   0), 6),
            'final_mlm_acc':    round(final_stats.get('mlm_acc',    0), 6),
            'final_nsp_acc':    round(final_stats.get('nsp_acc',    0), 6),

            # epoch별 상세 로그
            'epoch_logs': self.epoch_logs,
        }

    def _flatten(self, record: dict) -> dict:
        """CSV용: config dict와 epoch_logs를 제외한 평탄 dict"""
        flat = {}
        for k, v in record.items():
            if k == 'config':
                for ck, cv in v.items():
                    flat[f'cfg_{ck}'] = cv
            elif k == 'epoch_logs':
                pass   # CSV에는 포함하지 않음 (JSON에만 보관)
            else:
                flat[k] = v
        return flat

    # ── 터미널 출력 ──────────────────────────────────────────────

    def _print_summary(self, current: dict, previous_records: list):
        SEP = "=" * 64

        print(f"\n{SEP}")
        print(f"  실험 #{current['experiment_id']}  결과 요약  (mini BERT Pretrain)")
        print(SEP)

        print(f"  {'날짜/시간':<22} {current['timestamp'][:19]}")
        print(f"  {'소요 시간':<22} {current['duration_sec']} 초")
        print(f"  {'총 파라미터':<22} {current['total_params']:,}")
        print()

        cfg = current['config']
        print(f"  [하이퍼파라미터]")
        print(f"  {'vocab_size':<22} {cfg.get('vocab_size')}")
        print(f"  {'d_model':<22} {cfg.get('d_model')}")
        print(f"  {'num_heads':<22} {cfg.get('num_heads')}")
        print(f"  {'num_layers':<22} {cfg.get('num_layers')}")
        print(f"  {'d_ff':<22} {cfg.get('d_ff')}  (x{cfg.get('d_ff', 0) / max(1, cfg.get('d_model', 1)):.1f})")
        print(f"  {'max_seq_len':<22} {cfg.get('max_seq_len')}")
        print(f"  {'dropout':<22} {cfg.get('dropout')}")
        print(f"  {'learning_rate':<22} {cfg.get('learning_rate')}")
        print(f"  {'batch_size':<22} {cfg.get('batch_size')}")
        print(f"  {'epochs':<22} {cfg.get('epochs')}")
        print()

        print(f"  [학습 결과]")
        print(f"  {'best_epoch':<22} {current['best_epoch']} / {cfg.get('epochs')}")
        print(f"  {'best_total_loss':<22} {current['best_total_loss']}")
        print(f"  {'best_mlm_loss':<22} {current['best_mlm_loss']}")
        print(f"  {'best_nsp_loss':<22} {current['best_nsp_loss']}")
        print(f"  {'best_mlm_acc':<22} {current['best_mlm_acc']:.4f}")
        print(f"  {'best_nsp_acc':<22} {current['best_nsp_acc']:.4f}")
        print(f"  {'final_total_loss':<22} {current['final_total_loss']}")
        print(f"  {'final_mlm_acc':<22} {current['final_mlm_acc']:.4f}")
        print(f"  {'final_nsp_acc':<22} {current['final_nsp_acc']:.4f}")

        if not previous_records:
            print(f"\n  (첫 번째 실험 — 이전 비교 없음)")
            print(SEP)
            return

        prev = previous_records[-1]
        print(f"\n  --- 이전 실험 #{prev['experiment_id']} 대비 변화 ---")

        def delta(key, higher_is_better=False):
            c = current.get(key)
            p = prev.get(key)
            if c is None or p is None:
                return 'N/A'
            if c == p:
                return f'{c}  (변화없음)'
            diff     = c - p
            improved = (diff < 0) if not higher_is_better else (diff > 0)
            sign     = '+' if diff > 0 else ''
            tag      = '(개선)' if improved else '(악화)'
            return f'{c}  [{sign}{diff:+.6f} {tag}]'

        print(f"  {'best_total_loss':<22} {delta('best_total_loss')}")
        print(f"  {'best_mlm_loss':<22} {delta('best_mlm_loss')}")
        print(f"  {'best_nsp_loss':<22} {delta('best_nsp_loss')}")
        print(f"  {'best_mlm_acc':<22} {delta('best_mlm_acc', higher_is_better=True)}")
        print(f"  {'best_nsp_acc':<22} {delta('best_nsp_acc', higher_is_better=True)}")

        # 하이퍼파라미터 변경 비교
        prev_cfg = prev.get('config', {})
        curr_cfg = current.get('config', {})
        changed = [
            (k, prev_cfg.get(k), curr_cfg.get(k))
            for k in curr_cfg
            if prev_cfg.get(k) != curr_cfg.get(k)
        ]
        if changed:
            print(f"\n  [하이퍼파라미터 변경]")
            for k, old, new in changed:
                print(f"    {k:<24} {old}  -->  {new}")
        else:
            print(f"\n  [하이퍼파라미터] 이전 실험과 동일")

        print(SEP)

    # ── 시각화 ──────────────────────────────────────────────────

    def _generate_plots(self, all_records: list):
        if not _PLOT_AVAILABLE:
            print("[ExperimentTracker] matplotlib 없음 — 그래프 생성 건너뜀")
            return

        self._plot_epoch_curves()
        self._plot_experiments_comparison(all_records)
        self._plot_heatmap(all_records)
        print(f"[ExperimentTracker] 그래프 저장 완료 → {MODEL_DIR}/")

    def _plot_epoch_curves(self):
        """현재 실험의 epoch별 loss / acc 곡선"""
        if not self.epoch_logs:
            return

        epochs     = [e['epoch']      for e in self.epoch_logs]
        tot_losses = [e['total_loss'] for e in self.epoch_logs]
        mlm_losses = [e['mlm_loss']   for e in self.epoch_logs]
        nsp_losses = [e['nsp_loss']   for e in self.epoch_logs]
        mlm_accs   = [e['mlm_acc']    for e in self.epoch_logs]
        nsp_accs   = [e['nsp_acc']    for e in self.epoch_logs]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 곡선
        ax = axes[0]
        ax.plot(epochs, tot_losses, 'o-', color='steelblue',   label='Total Loss')
        ax.plot(epochs, mlm_losses, 's-', color='darkorange',  label='MLM Loss')
        ax.plot(epochs, nsp_losses, '^-', color='green',       label='NSP Loss')
        ax.set_title(f'Experiment #{self.experiment_id} — Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy 곡선
        ax = axes[1]
        ax.plot(epochs, mlm_accs, 'o-', color='darkorange', label='MLM Acc')
        ax.plot(epochs, nsp_accs, 's-', color='green',      label='NSP Acc')
        ax.set_title(f'Experiment #{self.experiment_id} — Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle('mini BERT Pretraining', fontsize=13)
        fig.tight_layout()
        path = os.path.join(MODEL_DIR, f'exp{self.experiment_id:02d}_curves.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_experiments_comparison(self, all_records: list):
        """전체 실험 Best Total Loss 막대그래프"""
        if not all_records:
            return

        ids    = [r['experiment_id']   for r in all_records]
        losses = [r['best_total_loss'] for r in all_records]
        colors = ['steelblue'] * (len(ids) - 1) + ['tomato']

        fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.9 + 2), 5))
        bars = ax.bar([str(i) for i in ids], losses, color=colors)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
        ax.set_title('All Experiments — Best Total Loss')
        ax.set_xlabel('Experiment #')
        ax.set_ylabel('Best Total Loss')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(MODEL_DIR, 'experiments_comparison.png'), dpi=150)
        plt.close(fig)

    def _plot_heatmap(self, all_records: list):
        """하이퍼파라미터 vs Best Total Loss 히트맵"""
        if not all_records or not _PLOT_AVAILABLE:
            return

        cfg_keys = ['d_model', 'num_layers', 'num_heads', 'd_ff',
                    'dropout', 'learning_rate', 'batch_size']
        labels   = cfg_keys + ['best_total_loss']

        data, row_labels = [], []
        for r in all_records:
            cfg = r.get('config', {})
            row = [cfg.get(k) for k in cfg_keys] + [r.get('best_total_loss')]
            if any(v is None for v in row):
                continue
            data.append(row)
            row_labels.append(f"#{r['experiment_id']}")

        if not data:
            return

        arr = np.array(data, dtype=float)
        arr_norm = np.zeros_like(arr)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            mn, mx = col.min(), col.max()
            arr_norm[:, j] = (col - mn) / (mx - mn) if mx != mn else np.zeros_like(col)

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9 + 1),
                                        max(3, len(row_labels) * 0.5 + 1.5)))
        im = ax.imshow(arr_norm, aspect='auto', cmap='RdYlGn_r')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

        for i in range(len(row_labels)):
            for j in range(len(labels)):
                val = arr[i, j]
                if j == len(labels) - 1 or labels[j] in ('dropout', 'learning_rate'):
                    text = f'{val:.4f}'
                elif labels[j] in ('d_model', 'num_layers', 'num_heads',
                                   'd_ff', 'batch_size'):
                    text = f'{int(val)}'
                else:
                    text = f'{val:.2f}'
                brightness = arr_norm[i, j]
                color = 'white' if brightness < 0.2 or brightness > 0.8 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color=color)

        ax.set_title('Hyperparameter Heatmap (normalized)')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(MODEL_DIR, 'heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
