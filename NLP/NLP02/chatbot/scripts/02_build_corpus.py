"""
Step 3: KoNLPy Mecab을 사용한 코퍼스 구축
- Mecab 형태소 분석
- 어휘 사전 구축
- 토큰 저장
"""

import pandas as pd
import pickle
import os
import sys
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class CorpusBuilder:
    """Mecab을 사용한 코퍼스 구축 클래스"""
    
    def __init__(self):
        """초기화"""
        self._tokenizer_obj = None
        self.tokenizer_name = None

        # 1) Mecab 시도
        try:
            from konlpy.tag import Mecab
            self._tokenizer_obj = Mecab()
            self.tokenizer_name = 'Mecab'
            print("✓ Mecab loaded successfully")
        except Exception as e:
            print(f"✗ Mecab not available: {e}")

        # 2) Okt fallback
        if self._tokenizer_obj is None:
            try:
                from konlpy.tag import Okt
                self._tokenizer_obj = Okt()
                self.tokenizer_name = 'Okt'
                print("✓ Okt loaded as fallback tokenizer")
            except Exception as e:
                print(f"✗ Okt not available: {e}")

        # 3) whitespace fallback
        if self._tokenizer_obj is None:
            print("⚠️ Using whitespace tokenizer as fallback")
            self.tokenizer_name = 'whitespace'

        self.word_freq = Counter()
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.pos_list = []
        self.tokenized_sentences = []
    
    def tokenize(self, sentence):
        """
        형태소 분석기(Mecab/Okt/whitespace)를 사용하여 문장을 형태소로 분석

        Args:
            sentence (str): 분석할 문장

        Returns:
            list: 형태소 리스트
        """
        try:
            if self.tokenizer_name == 'whitespace':
                return sentence.split()
            return self._tokenizer_obj.morphs(sentence)
        except Exception as e:
            return sentence.split()
    
    def build_vocabulary(self, df, min_freq=2, max_vocab_size=None):
        """
        어휘 사전 구축
        
        Args:
            df (pd.DataFrame): 데이터
            min_freq (int): 최소 빈도 (이 이상인 단어만 포함)
            max_vocab_size (int): 최대 어휘 크기
        """
        print("\n[어휘 사전 구축]")
        print(f"Tokenizing {len(df)} sentences...")
        
        # 모든 문장 토큰화
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            q_tokens = self.tokenize(row['question'])
            a_tokens = self.tokenize(row['answer'])
            
            # 단어 빈도 계산
            self.word_freq.update(q_tokens)
            self.word_freq.update(a_tokens)
            
            # 토큰화된 문장 저장
            self.tokenized_sentences.append({
                'question_tokens': q_tokens,
                'answer_tokens': a_tokens
            })
        
        # 빈도 필터링
        filtered_words = [(word, freq) for word, freq in self.word_freq.items() 
                         if freq >= min_freq]
        filtered_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
        
        # 최대 어휘 크기 제한
        if max_vocab_size:
            filtered_words = filtered_words[:max_vocab_size]
        
        print(f"✓ Created vocabulary with {len(filtered_words)} words (min_freq={min_freq})")
        
        # word2idx, idx2word 구축
        for idx, (word, freq) in enumerate(filtered_words, start=4):  # 4부터 시작 (특수토큰 빈 공간 남김)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"✓ Vocabulary size: {len(self.word2idx)}")
        
        # 상위 단어 확인
        print("\n[상위 20개 단어 (빈도)]")
        for word, freq in filtered_words[:20]:
            print(f"  {word}: {freq}")
    
    def tokenize_to_indices(self):
        """
        토큰화된 문장을 인덱스로 변환
        
        Returns:
            list: 인덱싱된 문장 리스트
        """
        print("\n[문장을 인덱스로 변환]")
        indexed_sentences = []
        
        for sent_dict in tqdm(self.tokenized_sentences, total=len(self.tokenized_sentences)):
            q_indices = self._tokens_to_indices(sent_dict['question_tokens'])
            a_indices = self._tokens_to_indices(sent_dict['answer_tokens'])
            
            indexed_sentences.append({
                'question_idx': q_indices,
                'answer_idx': a_indices
            })
        
        return indexed_sentences
    
    def _tokens_to_indices(self, tokens):
        """
        토큰을 인덱스로 변환
        
        Args:
            tokens (list): 토큰 리스트
            
        Returns:
            list: 인덱스 리스트
        """
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx['<unk>'])  # 미등록 단어는 <unk>로
        return indices
    
    def get_statistics(self):
        """코퍼스 통계 반환"""
        stats = {
            'vocab_size': len(self.word2idx),
            'total_sentences': len(self.tokenized_sentences),
            'avg_q_length': sum(len(s['question_tokens']) for s in self.tokenized_sentences) / len(self.tokenized_sentences),
            'avg_a_length': sum(len(s['answer_tokens']) for s in self.tokenized_sentences) / len(self.tokenized_sentences),
            'max_q_length': max(len(s['question_tokens']) for s in self.tokenized_sentences),
            'max_a_length': max(len(s['answer_tokens']) for s in self.tokenized_sentences),
        }
        return stats
    
    def save_corpus(self, output_path):
        """
        코퍼스를 pickle로 저장
        
        Args:
            output_path (str): 저장 경로
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        corpus_data = {
            'tokenized_sentences': self.tokenized_sentences,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'tokenizer_name': self.tokenizer_name,
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(corpus_data, f)
        
        print(f"\n✓ Corpus saved to {output_path}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Step 3: Mecab 코퍼스 구축")
    print("=" * 60)
    
    # 전처리된 데이터 로드
    if not os.path.exists(config.CLEANED_CSV):
        print(f"✗ Error: Processed data not found at {config.CLEANED_CSV}")
        print("Please run 01_preprocess.py first")
        return
    
    print(f"\nLoading preprocessed data from {config.CLEANED_CSV}...")
    df = pd.read_csv(config.CLEANED_CSV)
    print(f"✓ Loaded {len(df)} sentences")
    
    # 코퍼스 구축
    corpus_builder = CorpusBuilder()
    corpus_builder.build_vocabulary(df, min_freq=1, max_vocab_size=config.VOCAB_SIZE)
    
    # 통계 출력
    stats = corpus_builder.get_statistics()
    print("\n[코퍼스 통계]")
    print(f"Vocabulary size: {stats['vocab_size']}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Average question length: {stats['avg_q_length']:.2f}")
    print(f"Average answer length: {stats['avg_a_length']:.2f}")
    print(f"Max question length: {stats['max_q_length']}")
    print(f"Max answer length: {stats['max_a_length']}")
    
    # 토큰화된 문장을 인덱스로 변환 (선택사항)
    indexed_sentences = corpus_builder.tokenize_to_indices()
    print(f"✓ Indexed {len(indexed_sentences)} sentences")
    
    # 코퍼스 저장
    corpus_builder.save_corpus(config.CORPUS_PKL)
    print(f"\n✓ Corpus building completed successfully!")

if __name__ == "__main__":
    main()
