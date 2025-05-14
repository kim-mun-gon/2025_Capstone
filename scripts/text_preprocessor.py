# text_preprocessor.py
import re
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm

# 형태소 분석기 
okt = Okt()

# 최소 필수 불용어만 유지
stopwords = [
    '의','가','이','은','들','는','과','도','를','으로','자','에','와','한','하다',
    '그리고','하지만','때문에','그것','거기','저기','이런','저런','그런'
]

# lean_text: 문자열 하나를 정제 (predict에서 사용됨)
def clean_text(text: str) -> str:
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z!?., ]", "", str(text))
    text = re.sub(r"[ㅋㅎㅠㅜ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

#  preprocess: 데이터프레임 전체 정제 (train/test용)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['document'])
    df = df.copy()
    
    df.loc[:, 'document'] = df['document'].astype(str)
    df.loc[:, 'document'] = df['document'].str.replace(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z!?., ]", "", regex=True)
    df.loc[:, 'document'] = df['document'].str.replace(r"[ㅋㅎㅠㅜ]+", "", regex=True)
    df.loc[:, 'document'] = df['document'].str.replace(r"\s+", " ", regex=True)
    df.loc[:, 'document'] = df['document'].str.strip()
    
    df.loc[:, 'document'] = df['document'].replace('', np.nan)
    df = df.dropna(subset=['document'])
    
    return df

# tokenize_and_remove_stopwords: 형태소 분석 + 불용어 제거
def tokenize_and_remove_stopwords(series: pd.Series) -> list:
    result = []
    for sentence in tqdm(series, desc="🔄 Tokenizing"):
        tokens = okt.morphs(sentence, stem=True)
        cleaned = [word for word in tokens if word not in stopwords]
        result.append(cleaned)
    return result
