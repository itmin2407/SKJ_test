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
        return record

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
