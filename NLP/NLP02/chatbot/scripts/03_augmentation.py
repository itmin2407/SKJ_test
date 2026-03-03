"""
Step 4: Word2Vec 기반 데이터 증강
- ko.bin 모델을 사용하여 유사 문장 생성
- 데이터를 3배로 증강
- 증강된 데이터 저장
"""

import pandas as pd
import numpy as np
import os
import sys
from gensim.models import KeyedVectors
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class DataAugmenter:
    """Word2Vec 기반 데이터 증강 클래스"""
    
    def __init__(self, ko_bin_path):
        """
        초기화
        
        Args:
            ko_bin_path (str): ko.bin 모델 경로
        """
        self.ko_bin_path = ko_bin_path
        self.word_vectors = None
        self.load_model()
    
    def load_model(self):
        """Word2Vec 모델 로드

        - ko.bin 파일이 너무 크면 자동으로 스킵
        - 로드 중 오류가 나도 예외를 전달하지 않고 None으로 설정
        """
        if not os.path.exists(self.ko_bin_path):
            print(f"✗ Error: Model not found at {self.ko_bin_path}")
            print("Please download ko.bin from:")
            print("https://github.com/Kyubyong/wordvectors")
            raise FileNotFoundError(f"ko.bin not found at {self.ko_bin_path}")

        size_bytes = os.path.getsize(self.ko_bin_path)
        if size_bytes > 3_000_000_000:  # 3GB threshold
            print(f"⚠️ ko.bin file is very large ({size_bytes/1e9:.1f}GB); skipping load to avoid potential hang")
            self.word_vectors = None
            return

        print(f"Loading Word2Vec model from {self.ko_bin_path}...")
        try:
            # try several loading strategies to handle different ko.bin formats
            load_attempted = False

            # 1) binary word2vec format
            try:
                self.word_vectors = KeyedVectors.load_word2vec_format(self.ko_bin_path, binary=True)
                load_attempted = True
            except Exception as e1:
                print(f"binary load failed ({e1})")

            # 2) text word2vec format
            if not load_attempted:
                try:
                    self.word_vectors = KeyedVectors.load_word2vec_format(self.ko_bin_path, binary=False)
                    load_attempted = True
                except Exception as e2:
                    print(f"text load failed ({e2})")

            # 3) try latin1 encoding to avoid UnicodeDecodeError on weird headers
            if not load_attempted:
                try:
                    self.word_vectors = KeyedVectors.load_word2vec_format(self.ko_bin_path, binary=False, encoding='latin1')
                    load_attempted = True
                except Exception as e3:
                    print(f"text+latin1 load failed ({e3})")

            # 4) try gensim native loader (in case file is a saved KeyedVectors/Model)
            if not load_attempted:
                try:
                    print("attempting KeyedVectors.load (gensim native format)...")
                    self.word_vectors = KeyedVectors.load(self.ko_bin_path, mmap='r')
                    load_attempted = True
                except Exception as e4:
                    print(f"KeyedVectors.load failed ({e4})")

            # 5) try FastText facebook model loader
            if not load_attempted:
                try:
                    from gensim.models.fasttext import load_facebook_model
                    print("attempting FastText facebook model loader...")
                    ft = load_facebook_model(self.ko_bin_path)
                    self.word_vectors = ft.wv
                    load_attempted = True
                except Exception as ft_err:
                    print(f"FastText load failed ({ft_err})")

            # 6) final fallback: try load_word2vec_format with mmap
            if not load_attempted:
                try:
                    print("final attempt: load_word2vec_format with mmap and binary heuristic...")
                    self.word_vectors = KeyedVectors.load_word2vec_format(self.ko_bin_path, binary=True, unicode_errors='ignore')
                    load_attempted = True
                except Exception as final_err:
                    print(f"final load attempt failed ({final_err})")
                    raise final_err

            print(f"✓ Model loaded successfully")
            try:
                print(f"Vocabulary size: {len(self.word_vectors)}")
            except Exception:
                pass
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            # don't raise so caller can handle fallback
            self.word_vectors = None
    
    def get_similar_words(self, word, topn=5):
        """
        단어와 유사한 단어들 가져오기
        
        Args:
            word (str): 단어
            topn (int): 반환할 유사 단어 개수
            
        Returns:
            list: (유사단어, 유사도) 튜플 리스트
        """
        try:
            if word in self.word_vectors:
                return self.word_vectors.most_similar(word, topn=topn)
            else:
                return []
        except Exception as e:
            return []
    
    def augment_sentence(self, sentence, num_augmentations=1, replace_ratio=0.3):
        """
        문장 증강 (단어 대체를 통한 변형)
        
        Args:
            sentence (str): 증강할 문장
            num_augmentations (int): 생성할 증강 문장 개수
            replace_ratio (float): 단어 변경 비율 (0.0~1.0)
            
        Returns:
            list: 증강된 문장 리스트
        """
        augmented_sentences = []
        words = sentence.split()
        
        if len(words) == 0:
            return [sentence]  # 빈 문장은 그대로 반환
        
        for _ in range(num_augmentations):
            augmented_words = words.copy()
            
            # 변경할 단어 개수 결정
            num_replacements = max(1, int(len(words) * replace_ratio))
            
            # 랜덤하게 단어 선택
            indices_to_replace = np.random.choice(
                len(words), 
                size=min(num_replacements, len(words)), 
                replace=False
            )
            
            for idx in indices_to_replace:
                word = augmented_words[idx]
                similar_words = self.get_similar_words(word, topn=3)
                
                if similar_words:
                    # 유사 단어 선택 (유사도가 높은 것부터 선택하되, 일부 랜덤성 추가)
                    chosen_word = similar_words[np.random.randint(0, len(similar_words))][0]
                    augmented_words[idx] = chosen_word
            
            augmented_sentence = ' '.join(augmented_words)
            if augmented_sentence != sentence:  # 원본과 다른 경우만 추가
                augmented_sentences.append(augmented_sentence)
        
        return augmented_sentences if augmented_sentences else [sentence]
    
    def augment_dataset(self, df, augmentation_factor=3):
        """
        데이터셋 전체 증강
        
        Args:
            df (pd.DataFrame): 원본 데이터
            augmentation_factor (int): 증강 배수 (3배 증강 시 3)
            
        Returns:
            pd.DataFrame: 증강된 데이터
        """
        # if model was not loaded, skip augmentation entirely
        if self.word_vectors is None:
            print("⚠️ word_vectors not available; skipping augmentation and returning original dataframe")
            return df.copy()

        augmented_data = []
        
        print(f"\nAugmenting dataset ({augmentation_factor}x)...")
        
        # 원본 데이터 추가
        augmented_data = df.copy()
        
        # 증강 데이터 생성
        augmentations_needed = (augmentation_factor - 1)  # 원본 1개 + 추가 생성
        
        for _ in range(augmentations_needed):
            print(f"Creating augmentation {_ + 1}/{augmentations_needed}...")
            aug_df = df.copy()
            
            aug_df['question'] = df['question'].apply(
                lambda x: self.augment_sentence(x, num_augmentations=1, replace_ratio=0.2)[0]
            )
            aug_df['answer'] = df['answer'].apply(
                lambda x: self.augment_sentence(x, num_augmentations=1, replace_ratio=0.2)[0]
            )
            
            augmented_data = pd.concat([augmented_data, aug_df], ignore_index=True)
        
        # 중복 제거
        augmented_data = augmented_data.drop_duplicates(
            subset=['question', 'answer'], 
            keep='first'
        )
        
        print(f"✓ Dataset augmentation completed")
        print(f"Original size: {len(df)}")
        print(f"Augmented size: {len(augmented_data)}")
        print(f"Increase: {len(augmented_data) / len(df):.2f}x")
        
        return augmented_data

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Step 4: 데이터 증강 (Word2Vec 기반)")
    print("=" * 60)
    
    # 전처리된 데이터 로드
    if not os.path.exists(config.CLEANED_CSV):
        print(f"✗ Error: Processed data not found at {config.CLEANED_CSV}")
        print("Please run 01_preprocess.py first")
        return
    
    print(f"\nLoading preprocessed data from {config.CLEANED_CSV}...")
    df = pd.read_csv(config.CLEANED_CSV)
    print(f"✓ Loaded {len(df)} sentences")
    
    # 증강 사용 여부 체크
    if not getattr(config, 'USE_AUGMENTATION', True):
        print("⚠️ config.USE_AUGMENTATION이 False로 설정되어 있어 증강을 건너뜁니다.")
        augmented_df = df.copy()
    else:
        # ko.bin 모델 확인
        if not os.path.exists(config.KO_BIN_MODEL):
            print(f"\n✗ Warning: ko.bin model not found at {config.KO_BIN_MODEL}")
            print("\n모델 다운로드 방법:")
            print("1. 다음 링크에서 ko.bin 다운로드:")
            print("   https://github.com/Kyubyong/wordvectors")
            print(f"2. {config.KO_BIN_MODEL}에 저장")
            print("\n또는 다음 명령어 실행:")
            print(f"mkdir -p {config.MODEL_PATH}")
            print(f"wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz")
            print(f"gunzip cc.ko.300.bin.gz -c > {config.KO_BIN_MODEL}")
            
            print("\n데이터 증강 없이 원본 데이터만 사용합니다.")
            augmented_df = df.copy()
        else:
            # 데이터 증강
            augmenter = DataAugmenter(config.KO_BIN_MODEL)
            augmented_df = augmenter.augment_dataset(
                df, 
                augmentation_factor=config.AUGMENTATION_FACTOR
            )
    
    # 샘플 확인
    print("\n[샘플 데이터 (처음 3개)]")
    print(augmented_df.head(3))
    
    # 통계
    print("\n[데이터 통계]")
    print(f"Question 평균 길이: {augmented_df['question'].str.len().mean():.2f}")
    print(f"Answer 평균 길이: {augmented_df['answer'].str.len().mean():.2f}")
    
    # 増强된 데이터 저장
    os.makedirs(os.path.dirname(config.AUGMENTED_CSV), exist_ok=True)
    augmented_df.to_csv(config.AUGMENTED_CSV, index=False, encoding='utf-8')
    print(f"\n✓ Augmented data saved to {config.AUGMENTED_CSV}")
    print(f"\n✓ Data augmentation completed successfully!")

if __name__ == "__main__":
    main()
