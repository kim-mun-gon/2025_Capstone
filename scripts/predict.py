# predict.py
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_preprocessor import tokenize_and_remove_stopwords, clean_text

# 1. 모델 및 토크나이저 로드
with open("models/tokenizer_config.pkl", "rb") as f:
    config = pickle.load(f)
tokenizer = config["tokenizer"]
max_len = config["max_len"]

model = load_model("models/sentiment_model.h5")

# 2. 예측 함수 정의
def predict_multi_column_csv(csv_path, output_path="predicted_result.csv"):
    df = pd.read_csv(csv_path)
    print(f"📂 파일 로드 완료 → 총 컬럼 수: {len(df.columns)}")

    result_dfs = []
    summary = {}

    for col in df.columns:
        comments = df[col].dropna().tolist()
        print(f"\n🎬 작품: {col} | 댓글 수: {len(comments)}")

        #  문자열 정제 + 형태소 분석 & 불용어 제거
        cleaned_comments = [clean_text(c) for c in comments]
        tokenized = tokenize_and_remove_stopwords(pd.Series(cleaned_comments))

        #  시퀀스화 및 패딩
        sequences = tokenizer.texts_to_sequences(tokenized)
        padded = pad_sequences(sequences, maxlen=max_len)

        #  예측 수행
        predictions = model.predict(padded)
        labels = ["긍정" if p >= 0.5 else "부정" for p in predictions]

        #  결과 저장
        result_df = pd.DataFrame({
            "작품명": col,
            "원문": comments,
            "예측 감정": labels
        })
        result_dfs.append(result_df)

        #  통계 출력
        pos = labels.count("긍정")
        neg = labels.count("부정")
        total = len(labels)
        summary[col] = f"긍정: {pos / total:.2%} | 부정: {neg / total:.2%}"

        print(f"📊 긍정: {pos} / 부정: {neg} → 비율: {summary[col]}")

    #  전체 결과 저장
    final_df = pd.concat(result_dfs, ignore_index=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["긍/부정 비율"])
    summary_df.to_csv("prediction_summary.csv", encoding="utf-8-sig")

    print(f"\n 예측 결과 저장 완료 → {output_path}, prediction_summary.csv")


# 3. 예시 실행
predict_multi_column_csv("웹소설/로맨스_댓글.csv")

