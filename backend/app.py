from flask import Flask, jsonify
import requests
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from apscheduler.schedulers.background import BackgroundScheduler
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

app = Flask(__name__)
CORS(app)

# 전역 캐시 데이터
naver_ranking_cache = []
teen_ranking_cache = []
twenty_ranking_cache = []
thirty_ranking_cache = []
forty_ranking_cache = []

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def get_tags():
    url = "https://novel.naver.com/webnovel/weekday"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    tags = [button.get_text(strip=True) for button in soup.select("button.tag")]
    return tags

@app.route("/api/tags")
def tags():
    return jsonify(get_tags())

def get_naver_webnovel_rankings():
    url = "https://novel.naver.com/webnovel/weekday"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    data = []
    for item in soup.select("li.item")[:15]:
        try:
            title = item.select_one("span.title").text.strip()
            author = item.select_one("span.author").text.strip()
            genre = item.select_one("span.genre").text.strip()
            count = item.select_one("span.count").text.strip()
            image = item.select_one("img")["src"]
            link = "https://novel.naver.com" + item.select_one("a")["href"]

            data.append({
                "title": title,
                "author": author,
                "genre": genre,
                "count": count,
                "image": image,
                "link": link
            })
        except Exception as e:
            print(f"[⚠️] 메인 랭킹 항목 파싱 오류: {e}")
            continue

    return data

def get_age_ranking(age_label):
    driver = create_driver()
    try:
        driver.get("https://novel.naver.com/webnovel/weekday")

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#agePopularTabPanel .button_tab"))
        )

        buttons = driver.find_elements(By.CSS_SELECTOR, "#agePopularTabPanel .button_tab")
        found = False
        for btn in buttons:
            text = btn.get_attribute("innerText").strip()
            print(f"[DEBUG] 버튼 텍스트: '{text}' vs 기대값: '{age_label}'")
            if text == age_label:
                driver.execute_script("arguments[0].click();", btn)
                found = True
                break

        if not found:
            print(f"[❌] '{age_label}' 버튼을 찾을 수 없습니다.")
            return []

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#agePopularRanking li.item"))
            )
        except TimeoutException:
            print(f"[⏰] 콘텐츠 로딩 시간 초과: '{age_label}' 탭 클릭 후 콘텐츠 안 나옴")
            return []

        soup = BeautifulSoup(driver.page_source, "html.parser")
        items = soup.select("#agePopularRanking li.item")

        data = []
        for item in items[:15]:
            try:
                title = item.select_one("span.title").text.strip()
                author = item.select_one("span.author").text.strip()
                genre = item.select_one("span.genre").text.strip()
                count = item.select_one("span.count").text.strip()
                image = item.select_one("img")["src"]
                link = "https://novel.naver.com" + item.select_one("a")["href"]

                data.append({
                    "title": title,
                    "author": author,
                    "genre": genre,
                    "count": count,
                    "image": image,
                    "link": link
                })
            except Exception as e:
                print(f"[⚠️] 연령대 항목 파싱 오류: {e}")
                continue

        return data

    except Exception as e:
        print(f"[🔥] 크롤링 중 에러 발생: {e}")
        return []

    finally:
        driver.quit()

def update_all_rankings():
    global naver_ranking_cache, teen_ranking_cache, twenty_ranking_cache, thirty_ranking_cache, forty_ranking_cache
    print("[🔄] 랭킹 데이터 업데이트 중...")
    naver_ranking_cache = get_naver_webnovel_rankings()
    teen_ranking_cache = get_age_ranking("10대")
    twenty_ranking_cache = get_age_ranking("20대")
    thirty_ranking_cache = get_age_ranking("30대")
    forty_ranking_cache = get_age_ranking("40대")
    print("[✅] 랭킹 데이터 업데이트 완료!")

@app.route("/api/naver-rankings")
def naver_rankings():
    return jsonify(naver_ranking_cache)

@app.route("/api/rankings/teen")
def teen_rankings():
    return jsonify(teen_ranking_cache)

@app.route("/api/rankings/twenty")
def twenty_rankings():
    return jsonify(twenty_ranking_cache)

@app.route("/api/rankings/thirty")
def thirty_rankings():
    return jsonify(thirty_ranking_cache)

@app.route("/api/rankings/forty")
def forty_rankings():
    return jsonify(forty_ranking_cache)

if __name__ == "__main__":
    update_all_rankings()
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_all_rankings, 'interval', minutes=5)
    scheduler.start()
    app.run(host="0.0.0.0", port=5001)
