from flask import Flask, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

def get_tags():
    url = "https://novel.naver.com/webnovel/weekday"
    headers = { "User-Agent": "Mozilla/5.0" }
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    tags = [button.get_text(strip=True) for button in soup.select("button.tag")]
    return tags

@app.route("/api/tags")
def tags():
    return jsonify(get_tags())

def get_naver_webnovel_rankings():
    url = "https://novel.naver.com/webnovel/weekday"
    headers = { "User-Agent": "Mozilla/5.0" }
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    data = []
    for item in soup.select("li.item")[:15]:  # 상위 5개만 가져오기
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
        except:
            continue

    return data

@app.route("/api/naver-rankings")
def rankings():
    return jsonify(get_naver_webnovel_rankings())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
