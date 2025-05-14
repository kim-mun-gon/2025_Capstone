import os
import re
import pickle
import joblib
import unicodedata
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from konlpy.tag import Okt
import torch.nn.functional as F
from kiwipiepy import Kiwi
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import json

# app.py 최상단에 한번만 선언
kiwi     = Kiwi()
kw_model = KeyBERT()

okt = Okt()

# ===== 전역 캐시 및 모델 로딩 =====
naver_ranking_cache, teen_ranking_cache = [], []
twenty_ranking_cache, thirty_ranking_cache, forty_ranking_cache = [], [], []

# 유사도 모델
similarity_model = SentenceTransformer("nlpai-lab/KURE-v1")
with open("./models/웹소설_유사도_분석모델_통합.pkl", "rb") as f:
    embedding_package = pickle.load(f)

# 감성 분석 파이프라인 로딩
# 새로운 감정 분석 파이프라인 로드
# 1) KR-BERT 파이프라인만 로드
with open("models/krbert_pipeline.pkl", "rb") as f:
    krbert_pipe = pickle.load(f)

# 2) HuggingFace AutoTokenizer 생성
#    krbert_pipe["model_name"] 에 저장된 모델 이름을 그대로 사용
tokenizer = AutoTokenizer.from_pretrained(krbert_pipe["model_name"])
max_len   = krbert_pipe["max_len"]

# 3) PyTorch용 KR-BERT 분류 모델 로딩
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    krbert_pipe["model_name"],
    num_labels=2
)
sentiment_model.load_state_dict(
    torch.load("models/krbert_dropout_30.pt", map_location="cpu")
)
sentiment_model.eval()

# 설정 로드
with open("models/impact_scorer_ver3.pkl", "rb") as f:
    cfg = pickle.load(f)

motifs        = cfg["motif_list"]
turn_keywords = cfg["turn_keywords"]
senti_dict    = cfg["sentiword_dict"]

def tokenize(text: str):
    if not text:
        return []
    return [t.form for t in kiwi.analyze(text)[0][0]]

def emotion_score(tokens):
    n = max(len(tokens), 1)
    hits = [w for w in tokens if senti_dict.get(w, 0) != 0]
    return min(len(hits) / n, 1.0), hits

def motif_score(text: str):
    n = max(len(motifs), 1)
    hits = [m for m in motifs if m in text]
    return min(len(hits) / n, 1.0), hits

def transition_score(text: str):
    n = max(len(turn_keywords), 1)
    hits = [kw for kw in turn_keywords if kw in text]
    return min(len(hits) / n, 1.0), hits



# 성공 예측 모델 로드
xgb_model = joblib.load("models/kure_xgb_total_model.pkl")
embed_model = SentenceTransformer("nlpai-lab/KURE-v1")

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
                "title":  item.select_one("span.title").text.strip(),
                "author": item.select_one("span.author").text.strip(),
                "genre":  item.select_one("span.genre").text.strip(),
                "count":  item.select_one("span.count").text.strip(),
                "image":  item.select_one("img")["src"],
                "link":   "https://novel.naver.com" + item.select_one("a")["href"]
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
                    "title":  item.select_one("span.title").text.strip(),
                    "author": item.select_one("span.author").text.strip(),
                    "genre":  item.select_one("span.genre").text.strip(),
                    "count":  item.select_one("span.count").text.strip(),
                    "image":  item.select_one("img")["src"],
                    "link":   "https://novel.naver.com" + item.select_one("a")["href"]
                })
            except:
                continue
        return data
    except:
        return []
    finally:
        driver.quit()

def update_all_rankings():
    global naver_ranking_cache, teen_ranking_cache
    global twenty_ranking_cache, thirty_ranking_cache, forty_ranking_cache

    naver_ranking_cache  = get_naver_webnovel_rankings()
    teen_ranking_cache   = get_age_ranking("10대")
    twenty_ranking_cache = get_age_ranking("20대")
    thirty_ranking_cache = get_age_ranking("30대")
    forty_ranking_cache  = get_age_ranking("40대")

def parse_count(text):
    text = re.sub(r"[가-힣\s,]*", "", str(text))
    m = re.search(r"(\d+)(만)?", text)
    if not m:
        return 0
    return int(m.group(1)) * 10000 if m.group(2) else int(m.group(1))

@asynccontextmanager
async def lifespan(app: FastAPI):
    update_all_rankings()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class SummaryRequest(BaseModel):
    title: str
    genre: str
    summary: str

class PredictRequest(BaseModel):
    title: str
    summary: str
    genre: str

class SuccessRequest(BaseModel):
    title: str
    genre: str
    summary: str

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

@app.post("/api/similarity")
def get_similar_novels(req: SummaryRequest):
    genre_norm = unicodedata.normalize("NFC", req.genre.strip())
    if genre_norm not in embedding_package:
        return {"error": f"'{req.genre}' 장르 데이터 없음"}

    vec    = similarity_model.encode(f"{req.title} {req.summary}")
    data   = embedding_package[genre_norm]
    scores = util.cos_sim(vec, data["embeddings"])[0]
    idxs   = np.argsort(-scores)[:10]

    similar_titles = [data["titles"][i] for i in idxs]

    return {
        "similar_novels": [
            {
                "title":      data["titles"][i],
                "summary":    data["summaries"][i],
                "similarity": f"{scores[i].item()*100:.1f}%"
            }
            for i in idxs
        ],
        "similar_titles": similar_titles
    }

@app.post("/api/predict-score")
def predict_score(req: PredictRequest):
    genre_norm = unicodedata.normalize("NFC", req.genre.strip())
    if genre_norm not in embedding_package:
        return {"error": f"'{req.genre}' 장르 데이터 없음"}

    vec    = similarity_model.encode(f"{req.title} {req.summary}")
    data   = embedding_package[genre_norm]
    scores = util.cos_sim(vec, data["embeddings"])[0]
    idxs   = np.argsort(-scores)[:10]
    similar_titles = [data["titles"][i] for i in idxs]

    info_path    = f"webnovel_data/{genre_norm}_기본정보_줄거리포함.csv"
    comment_path = f"webnovel_data/{genre_norm}_댓글.csv"
    if not os.path.exists(info_path) or not os.path.exists(comment_path):
        return {"error": "필요한 데이터 파일이 없습니다."}

    df = pd.read_csv(info_path)
    df = df[df["제목"].isin(similar_titles)]
    comments = pd.read_csv(comment_path)
    comments = comments[similar_titles]

    df["다운로드 수"] = df["다운로드 수"].apply(parse_count)
    df["관심 수"]   = df["관심 수"].apply(parse_count)

    from text_preprocessor import clean_text, tokenize_and_remove_stopwords

    pos, neg = {}, {}
    for col in comments.columns:
        texts = comments[col].dropna().tolist()
        if not texts:
            pos[col] = neg[col] = np.nan
            continue

        # 1) 전처리
        cleaned = [clean_text(t) for t in texts]

        # 2) HuggingFace 토크나이저로 배치 인코딩
        enc = tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        # 3) BERT inference
        with torch.no_grad():
            outputs = sentiment_model(**enc)
            probs   = F.softmax(outputs.logits, dim=1).cpu().numpy()

        # 4) 긍정 클래스(preds[:,1]) 기준으로 이진화
        preds = (probs[:, 1] >= 0.5).astype(int)

        # 5) 비율 계산
        total     = len(preds)
        pos[col]  = round(preds.sum() / total, 5)
        neg[col]  = round((total - preds.sum()) / total, 5)

    df["긍정 비율"] = df["제목"].map(pos)
    df["부정 비율"] = df["제목"].map(neg)
    df = df.dropna(subset=["긍정 비율", "부정 비율", "별점"])
    if df.empty:
        return {"error": "충분한 데이터가 없습니다."}

    avg_positive = df["긍정 비율"].mean()
    w = {"다운로드 수":0.1, "관심 수":0.25, "긍정 비율":0.35, "부정 비율":0.2, "별점":0.1}
    norm = lambda x: x / x.max()
    weighted = (
        norm(df["다운로드 수"]).mean() * w["다운로드 수"]
      + norm(df["관심 수"]).mean()   * w["관심 수"]
      + df["긍정 비율"].mean()       * w["긍정 비율"]
      + df["부정 비율"].mean()       * w["부정 비율"]
      + (df["별점"].mean() / 10)     * w["별점"]
    ) * 10

    Xs = MinMaxScaler().fit_transform(df[["다운로드 수","관심 수","긍정 비율","부정 비율","별점"]])
    rf = RandomForestRegressor(random_state=42)
    rf.fit(Xs, df["별점"])
    ml_pred = rf.predict([Xs.mean(axis=0)])[0]

    alpha = 0.7 if len(df) < 10 else 0.5 if len(df) < 20 else 0.3
    hybrid = weighted * alpha + ml_pred * (1 - alpha)

    return {
        "predicted_rating": round(hybrid, 2),
        "positive_rate":     round(avg_positive, 5)
    }

@app.post("/api/success-predict")
def success_predict(req: SuccessRequest):
    # 1) 유사작품 5개 추출
    genre_norm = unicodedata.normalize("NFC", req.genre.strip())
    if genre_norm not in embedding_package:
        return {"error": "장르 데이터가 없습니다."}

    # similarity_model + embedding_package 에서 titles, embeddings 불러오기
    vec_scores = similarity_model.encode(f"{req.title} {req.summary}")
    data       = embedding_package[genre_norm]
    scores     = util.cos_sim(vec_scores, data["embeddings"])[0]
    idxs       = np.argsort(-scores)[:5]
    similar_titles = [data["titles"][i] for i in idxs]

    # 2) 데이터 로드 & 유사작품 필터링
    info_path = f"webnovel_data/{genre_norm}_기본정보_줄거리포함.csv"
    if not os.path.exists(info_path):
        return {"error": "장르 데이터가 없습니다."}

    df     = pd.read_csv(info_path)
    df_sim = df[df["제목"].isin(similar_titles)]
    if df_sim.empty:
        return {"error": "충분한 유사작 데이터가 없습니다."}

    # 3) 유사작 기반 지표 계산
    sim_avg    = df_sim["별점"].mean()
    dl_avg     = df_sim["다운로드 수"] \
                     .apply(lambda x: float(re.sub(r"[^\d.]", "", str(x)) or 0)) \
                     .mean()
    it_avg     = df_sim["관심 수"] \
                     .apply(lambda x: int(re.sub(r"[^\d]", "", str(x)) or 0)) \
                     .mean()
    genre_code = pd.Series([req.genre]).astype("category").cat.codes[0]

    # 4) XGBoost 입력 벡터 준비 (1-D numpy arrays)
    # 4-1) 줄거리 임베딩 벡터
    embed_vec = embed_model.encode([req.summary])
    embed_vec = np.array(embed_vec[0], dtype=float)   # (dim,)

    # 4-2) 피쳐 벡터
    feat_vec  = np.array([dl_avg, it_avg, genre_code], dtype=float)  # (3,)

    # 5) 최종 입력 벡터 결합
    input_vec  = np.concatenate((embed_vec, feat_vec))  # (dim+3,)

    # 6) 예측 평점 산출
    exp_rating  = float(xgb_model.predict([input_vec])[0])

    # 7) 성공 확률 가중 평균 (모두 유사작 기반)
    success_prob = exp_rating * 0.5 + sim_avg * 0.3 + sim_avg * 0.2

    # 8) 등급 매핑
    if success_prob >= 9.5:
        grade = "S"
    elif success_prob >= 9.0:
        grade = "A"
    elif success_prob >= 8.0:
        grade = "B"
    elif success_prob >= 7.0:
        grade = "C"
    else:
        grade = "D"

    # 9) 정성 피드백
    feedback = []
    if exp_rating < 6.5:
        feedback.append("🛠️ 스토리 개연성 및 몰입도를 강화해보세요.")
    if exp_rating > 9.0 and sim_avg < 8.0:
        feedback.append("🧪 실험적인 구성 고려: 시장 검증이 필요한 요소입니다.")
    if sim_avg >= 9.2 and exp_rating < sim_avg - 1:
        feedback.append("📊 유사작 평균 평점이 높아 경쟁이 치열합니다. 차별화된 설정이 필요합니다.")
    if hasattr(req, "positive_rate") and req.positive_rate < 0.4:
        feedback.append("🗨️ 유사 작품들의 댓글에서 부정적 의견이 많습니다. 구성 개선을 검토해보세요.")
    if not feedback:
        feedback.append("✅ 기본 요소 양호하나, 독창적 전개를 추가해보세요.")

    return {
        "success_probability": round(success_prob, 2),
        "grade":               grade,
        "feedback":            feedback
    }



@app.post("/api/novelty-score")
def get_novelty_score(req: SummaryRequest):
    import pickle

    # 1) 모델 데이터 로드 (장르 키워드, 불용어)
    with open("models/novelty_model_v5.pkl", "rb") as f:
        model_data = pickle.load(f)
    genre_keywords = model_data["genre_keywords"]
    stopwords      = set(model_data["stopwords"])

    # 2) 키워드 추출 함수
    def extract_keywords(text, top_n=100):
        allowed_pos = {"NNG", "NNP"}
        tokens = [
            token for token, pos, _, _
            in kiwi.analyze(text)[0][0]
            if pos in allowed_pos and len(token) > 1 and token not in stopwords
        ]
        joined = " ".join(tokens)
        kws = kw_model.extract_keywords(
            joined,
            keyphrase_ngram_range=(1, 1),
            use_mmr=True,
            diversity=0.5,
            top_n=top_n
        )
        return [kw for kw, _ in kws]

    # 3) 사용자 키워드 & 장르 키워드 비교
    user_kws = extract_keywords(req.summary)
    user_set = set(user_kws)
    genre_set = set(genre_keywords.get(req.genre, []))

    overlap_ratio = len(user_set & genre_set) / (len(user_set) or 1)
    score = round((1 - overlap_ratio) * 100, 1)

    return {
        "novelty_score": score,
        "user_keywords": user_kws,
        "overlap_keywords": list(user_set & genre_set),
        "overlap_ratio": f"{round(overlap_ratio*100,1)}%"
    }


@app.post("/api/impact-score")
def get_impact_score(req: SummaryRequest):
    text = req.summary or ""

    # 1) 토큰화 및 각 요소 히트
    tokens    = tokenize(text)
    _, emo_h  = emotion_score(tokens)
    _, mot_h  = motif_score(text)
    _, trn_h  = transition_score(text)

    # 2) 최종 임팩트 점수 (비율은 내부만 사용)
    emo_s, _  = emotion_score(tokens)
    mot_s, _  = motif_score(text)
    trn_s, _  = transition_score(text)
    impact_score = round((emo_s * 0.4 + mot_s * 0.3 + trn_s * 0.3) * 10, 2)

    # 3) 반환
    return {
        "impact_score":    impact_score,
        "emotion_hits":    emo_h,
        "motif_hits":      mot_h,
        "transition_hits": trn_h
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
