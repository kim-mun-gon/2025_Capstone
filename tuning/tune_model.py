# tune_model.py (train 구조 기반 튜닝 버전)

import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from text_preprocessor import preprocess, tokenize_and_remove_stopwords

# 1. 데이터 로딩
train_df = pd.read_table("ratings_train.txt")
custom_df = pd.read_csv("final_tune_set.txt", sep="\t", engine="python")  # 커스텀 튜닝셋
train_data = pd.concat([train_df[['document', 'label']], custom_df], ignore_index=True)
train_data = preprocess(train_data)  # 정제

# 2. 토큰화
X = tokenize_and_remove_stopwords(train_data['document'])
y = train_data['label']

# 3. Tokenizer 로드
with open("models/tokenizer_config.pkl", "rb") as f:
    config = pickle.load(f)
tokenizer = config["tokenizer"]
max_len = config["max_len"]

# 4. 시퀀스 변환 및 비어있는 샘플 제거
X_seq = tokenizer.texts_to_sequences(X)
X_y_pair = [(x, y_) for x, y_ in zip(X_seq, y) if len(x) > 0]
X_seq, y = zip(*X_y_pair)
X_pad = pad_sequences(X_seq, maxlen=max_len)
y = np.array(y)

# 5. 모델 재정의 (train 구조 그대로 사용)
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 200))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0003), loss="binary_crossentropy", metrics=["accuracy"])

# 6. 학습
early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
model.fit(X_pad, y, validation_split=0.2, epochs=10, batch_size=32, callbacks=[early_stopping])

# 7. 저장
model.save("models/sentiment_model.h5")
print("\n 튜닝 완료! 모델 재저장됨.")
