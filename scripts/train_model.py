import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional,
    Dense, Dropout, Embedding, LSTM
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. NSMC 데이터 다운로드 및 로딩
os.makedirs("data", exist_ok=True)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
    "data/ratings_train.txt"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
    "data/ratings_test.txt"
)
train_data = pd.read_table("data/ratings_train.txt", encoding="utf-8")
test_data  = pd.read_table("data/ratings_test.txt", encoding="utf-8")

# 2. 전처리
def preprocess(df):
    df = df.dropna()
    df = df[df["document"].str.strip() != ""]
    return df

train_data = preprocess(train_data)
test_data  = preprocess(test_data)

# 3. 토크나이징 및 불용어 제거
okt = Okt()
stopwords = set([
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걍',
    '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
    '고', '에서', '까지'
])

def tokenize_and_remove_stopwords(text_list):
    tokenized = []
    for sentence in text_list:
        try:
            tokens = okt.morphs(str(sentence), stem=True)
            clean = [w for w in tokens if w not in stopwords and len(w) > 1]
            tokenized.append(clean)
        except Exception:
            tokenized.append([])
    return tokenized

X_train_tokens = tokenize_and_remove_stopwords(train_data["document"])
X_test_tokens  = tokenize_and_remove_stopwords(test_data["document"])

# 4. 라벨 정리 및 빈 토큰 제거
train_pairs = [(x, y) for x, y in zip(X_train_tokens, train_data["label"]) if len(x) > 0]
X_train, y_train = zip(*train_pairs)
y_train = np.array(y_train)

test_pairs = [(x, y) for x, y in zip(X_test_tokens, test_data["label"]) if len(x) > 0]
X_test, y_test = zip(*test_pairs)
y_test = np.array(y_test)

# 5. 시퀀싱 및 패딩
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)

lengths = [len(seq) for seq in X_train_seq]
max_len = int(np.percentile(lengths, 95))
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

# 6. 모델 정의 (Input → shape만 지정, batch_shape X)
input_layer = keras.Input(shape=(max_len,), name="input")
x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=200)(input_layer)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(64))(x)
x = BatchNormalization()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 7. 학습
early_stopping = EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)
model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stopping]
)

# 8. 평가
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
y_pred = model.predict(X_test_pad, verbose=1)
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n[테스트 성능]\n")
print(classification_report(y_test, y_pred_labels, target_names=["부정", "긍정"]))

# 9. 모델, 토크나이저, max_len 저장 (✅ batch_shape 없음)
pipeline = {
    "model_json": model.to_json(),  # batch_shape 없이 저장됨
    "model_weights": model.get_weights(),
    "tokenizer": tokenizer,
    "max_len": max_len
}

os.makedirs("models", exist_ok=True)
with open("models/sentiment_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ 통합 PKL 저장 완료 → models/sentiment_pipeline.pkl")
