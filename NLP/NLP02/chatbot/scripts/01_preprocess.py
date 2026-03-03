"""
Step 1-2: 챗봇 데이터 전처리 스크립트
- CSV 파일 읽기
- 정규식을 사용한 데이터 정제
- 정규화 및 정제된 데이터 저장
"""

import pandas as pd
import re
import os
from pathlib import Path
import sys

# config 파일 임포트를 위해 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def preprocess_sentence(sentence):
    """
    정규식을 사용하여 문장을 정제하는 함수
    
    Args:
        sentence (str): 정제할 문장
        
    Returns:
        str: 정제된 문장
    """
    if not isinstance(sentence, str):
        return ""
    
    # 공백 정규화 (여러 공백을 하나로)
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # 특수문자 제거 (한글, 영문, 숫자, 기본 구두점만 유지)
    sentence = re.sub(r'[^\w\s가-힣.!?]', '', sentence)
    
    # URL 제거
    sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence)
    
    # 연속된 구두점 정리 (예: "???" -> "?")
    sentence = re.sub(r'[.!?]{2,}', lambda m: m.group(0)[0], sentence)
    
    # 앞뒤 공백 제거
    sentence = sentence.strip()
    
    return sentence

def load_and_preprocess_data(csv_path):
    """
    CSV 파일을 로드하고 전처리하는 함수
    
    Args:
        csv_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: 전처리된 데이터
    """
    print(f"Loading data from {csv_path}...")
    
    try:
        # CSV 파일 읽기 (다양한 인코딩 시도)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='cp949')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='euc-kr')
        
        print(f"✓ Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
    except FileNotFoundError:
        print(f"✗ Error: File {csv_path} not found")
        return None
    
    # 데이터 프로퍼티 확인
    print("\n[데이터 정보]")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 결측값 확인
    print(f"\n[결측값 현황]")
    print(df.isnull().sum())
    
    # 첫 번째 열과 두 번째 열을 question, answer로 가정
    # (실제 컬럼명이 다르면 수정 필요)
    columns = df.columns.tolist()
    
    if len(columns) >= 2:
        question_col = columns[0]
        answer_col = columns[1]
    else:
        print("✗ Error: CSV must have at least 2 columns")
        return None
    
    print(f"\nQuestion column: {question_col}, Answer column: {answer_col}")
    
    # 결측값 제거
    df = df.dropna(subset=[question_col, answer_col])
    print(f"\n✓ After removing NaN: {len(df)} rows")
    
    # 전처리 적용
    print("\n[전처리 진행중...]")
    df[question_col] = df[question_col].apply(preprocess_sentence)
    df[answer_col] = df[answer_col].apply(preprocess_sentence)
    
    # 빈 문장 제거
    df = df[(df[question_col].str.len() > 0) & (df[answer_col].str.len() > 0)]
    print(f"✓ After removing empty sentences: {len(df)} rows")
    
    # 중복 제거
    initial_len = len(df)
    df = df.drop_duplicates(subset=[question_col, answer_col])
    removed_duplicates = initial_len - len(df)
    print(f"✓ Removed {removed_duplicates} duplicates. Remaining: {len(df)} rows")
    
    # 컬럼명 표준화
    df = df.rename(columns={question_col: 'question', answer_col: 'answer'})
    
    return df

def save_processed_data(df, output_path):
    """
    전처리된 데이터를 CSV로 저장
    
    Args:
        df (pd.DataFrame): 저장할 데이터
        output_path (str): 저장 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Processed data saved to {output_path}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Step 1-2: 데이터 전처리")
    print("=" * 60)
    
    # CSV 파일 확인
    if not os.path.exists(config.CSV_FILE):
        print(f"✗ Error: CSV file not found at {config.CSV_FILE}")
        print(f"Please place ChatbotData.csv in {config.DATA_RAW_PATH}/")
        return
    
    # 데이터 로드 및 전처리
    df = load_and_preprocess_data(config.CSV_FILE)
    
    if df is not None:
        # 샘플 데이터 확인
        print("\n[샘플 데이터 (처음 5개)]")
        print(df.head())
        
        # 통계 정보
        print("\n[문장 길이 통계]")
        print(f"Question 평균 길이: {df['question'].str.len().mean():.2f}")
        print(f"Answer 평균 길이: {df['answer'].str.len().mean():.2f}")
        
        # 정제된 데이터 저장
        save_processed_data(df, config.CLEANED_CSV)
        print(f"\n✓ Preprocessing completed successfully!")
        print(f"Output file: {config.CLEANED_CSV}")
    else:
        print("\n✗ Preprocessing failed")

if __name__ == "__main__":
    main()
