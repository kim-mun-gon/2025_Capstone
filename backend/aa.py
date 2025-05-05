나의 말:
# app.py
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
import pickle
import unicodedata
import pandas as pd
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, BatchNormalization, Dense

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ===== 전역 캐시 및 모델 로딩 =====
naver_ranking_cache, teen_ranking_cache = [], []
twenty_ranking_cache, thirty_ranking_cache, forty_ranking_cache = [], [], []

similarity_model = SentenceTransformer("nlpai-lab/KURE-v1")
with open("./models/웹소설_유사도_분석모델_통합.pkl", "rb") as f:
    embedding_package = pickle.load(f)

with open("models/sentiment_pipeline.pkl", "rb") as f:
    sentiment_pipeline = pickle.load(f)

tokenizer = sentiment_pipeline["tokenizer"]
max_len   = sentiment_pipeline["max_len"]
vocab_size = len(tokenizer.word_index) + 1

# ===== 감성 모델 구조 재구성 및 가중치 로드 =====
inputs = keras.Input(shape=(max_len,), name="input")
x = Embedding(input_dim=vocab_size, output_dim=200)(inputs)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(64))(x)
x = BatchNormalization()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

sentiment_model = Model(inputs=inputs, outputs=outputs)
sentiment_model.compile(optimizer=Adam(3e-4), loss="binary_crossentropy", metrics=["accuracy"])
sentiment_model.set_weights(sentiment_pipeline["model_weights"])

# ===== 유틸 함수 =====
def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def get_tags():
    res = requests.get("https://novel.naver.com/webnovel/weekday", headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    return [btn.get_text(strip=True) for btn in soup.select("button.tag")]

def get_naver_webnovel_rankings():
    res = requests.get("https://novel.naver.com/webnovel/weekday", headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    data = []
    for item in soup.select("li.item")[:15]:
        try:
            data.append({
                "title": item.select_one("span.title").text.strip(),
                "author": item.select_one("span.author").text.strip(),
                "genre": item.select_one("span.genre").text.strip(),
                "count": item.select_one("span.count").text.strip(),
                "image": item.select_one("img")["src"],
                "link": "https://novel.naver.com" + item.select_one("a")["href"]
            })
        except:
            continue
    return data

def get_age_ranking(age_label):
    driver = create_driver()
    try:
        driver.get("https://novel.naver.com/webnovel/weekday")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#agePopularTabPanel .button_tab"))
        )
        tabs = driver.find_elements(By.CSS_SELECTOR, "#agePopularTabPanel .button_tab")
        for tab in tabs:
            if tab.get_attribute("innerText").strip() == age_label:
                driver.execute_script("arguments[0].scrollIntoView(true);", tab)
                driver.execute_script("arguments[0].click();", tab)
                break
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#agePopularRanking li.item"))
        )
        soup = BeautifulSoup(driver.page_source, "html.parser")
        data = []
        for item in soup.select("#agePopularRanking li.item")[:15]:
            try:
                data.append({
                    "title": item.select_one("span.title").text.strip(),
                    "author": item.select_one("span.author").text.strip(),
                    "genre": item.select_one("span.genre").text.strip(),
                    "count": item.select_one("span.count").text.strip(),
                    "image": item.select_one("img")["src"],
                    "link": "https://novel.naver.com" + item.select_one("a")["href"]
                })
            except:
                continue
        return data
    except (TimeoutException, ElementClickInterceptedException) as e:
        print(f"[🔥] 연령대 '{age_label}' 크롤링 오류: {e}")
        return []
    finally:
        driver.quit()

def update_all_rankings():
    global naver_ranking_cache, teen_ranking_cache, twenty_ranking_cache, thirty_ranking_cache, forty_ranking_cache
    print("[🔄] 랭킹 업데이트 중...")
    naver_ranking_cache   = get_naver_webnovel_rankings()
    teen_ranking_cache    = get_age_ranking("10대")
    twenty_ranking_cache  = get_age_ranking("20대")
    thirty_ranking_cache  = get_age_ranking("30대")
    forty_ranking_cache   = get_age_ranking("40대")
    print("[✅] 랭킹 업데이트 완료")

def parse_count(text):
    text = re.sub(r"[가-힣\s,]*", "", str(text))
    m = re.search(r"(\d+)(만)?", text)
    return int(m.group(1)) * 10000 if m and m.group(2) else int(m.group(1)) if m else 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    update_all_rankings()
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_all_rankings, 'interval', minutes=5)
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/tags")
async def tags():
    return JSONResponse(content=get_tags())

@app.get("/api/naver-rankings")
async def naver_rankings():
    return JSONResponse(content=naver_ranking_cache)

@app.get("/api/rankings/teen")
async def teen_rankings():
    return JSONResponse(content=teen_ranking_cache)

@app.get("/api/rankings/twenty")
async def twenty_rankings():
    return JSONResponse(content=twenty_ranking_cache)

@app.get("/api/rankings/thirty")
async def thirty_rankings():
    return JSONResponse(content=thirty_ranking_cache)

@app.get("/api/rankings/forty")
async def forty_rankings():
    return JSONResponse(content=forty_ranking_cache)

class SummaryRequest(BaseModel):
    title: str
    genre: str
    summary: str

@app.post("/api/similarity")
def get_similar_novels(req: SummaryRequest):
    genre = unicodedata.normalize("NFC", req.genre.strip())
    if genre not in embedding_package:
        return {"error": f"'{genre}' 장르 데이터 없음"}
    vec = similarity_model.encode(f"{req.title} {req.summary}")
    data = embedding_package[genre]
    scores = util.cos_sim(vec, data["embeddings"])[0]
    idxs  = np.argsort(-scores)[:6]
    return {"similar_novels": [
        {"title": data["titles"][i], "summary": data["summaries"][i], "similarity": f"{scores[i].item()*100:.1f}%"}
        for i in idxs
    ]}

class PredictRequest(BaseModel):
    genre: str

@app.post("/api/predict-score")
def predict_score(req: PredictRequest):
    genre = req.genre.strip()
    info_path    = f"webnovel_data/{genre}_기본정보_줄거리포함.csv"
    comment_path = f"webnovel_data/{genre}_댓글.csv"
    if not os.path.exists(info_path) or not os.path.exists(comment_path):
        return {"error": "필요한 데이터 파일이 없습니다."}

    df       = pd.read_csv(info_path)
    comments = pd.read_csv(comment_path)
    df["다운로드 수"] = df["다운로드 수"].apply(parse_count)
    df["관심 수"]   = df["관심 수"].apply(parse_count)

    from text_preprocessor import clean_text, tokenize_and_remove_stopwords
    pos, neg = {}, {}
    for col in comments.columns:
        texts = comments[col].dropna().tolist()
        cleaned = [clean_text(t) for t in texts]
        toks = tokenize_and_remove_stopwords(pd.Series(cleaned))
        seq = tokenizer.texts_to_sequences(toks)
        pad = pad_sequences(seq, maxlen=max_len)
        if len(pad)==0:
            pos[col]=neg[col]=np.nan; continue
        preds = sentiment_model.predict(pad, verbose=0)
        b = (preds>=0.5).astype(int).flatten()
        total = len(b)
        pos[col]=round(b.sum()/total,5)
        neg[col]=round((total-b.sum())/total,5)

    df["긍정 비율"] = df["제목"].map(pos)
    df["부정 비율"] = df["제목"].map(neg)
    df = df.dropna(subset=["긍정 비율","부정 비율","별점"])
    if df.empty:
        return {"error": "충분한 데이터가 없습니다."}

    # 평균 긍정 비율
    avg_positive = df["긍정 비율"].mean()

    w = {"다운로드 수":0.1,"관심 수":0.25,"긍정 비율":0.35,"부정 비율":0.2,"별점":0.1}
    norm = lambda x:x/x.max()
    weighted = (
        norm(df["다운로드 수"]).mean()*w["다운로드 수"]
      + norm(df["관심 수"]).mean()*w["관심 수"]
      + df["긍정 비율"].mean()*w["긍정 비율"]
      + df["부정 비율"].mean()*w["부정 비율"]
      + (df["별점"].mean()/10)*w["별점"]
    ) *10

    Xs = MinMaxScaler().fit_transform(df[["다운로드 수","관심 수","긍정 비율","부정 비율","별점"]])
    rf = RandomForestRegressor(random_state=42)
    rf.fit(Xs, df["별점"])
    ml = rf.predict([Xs.mean(axis=0)])[0]
    alpha = 0.7 if len(df)<10 else 0.5 if len(df)<20 else 0.3
    hybrid = weighted*alpha + ml*(1-alpha)

    return {
        "predicted_rating": round(hybrid,2),
        "positive_rate": round(avg_positive,5)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)