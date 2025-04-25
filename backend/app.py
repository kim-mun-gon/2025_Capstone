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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
