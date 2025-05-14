import time
import urllib.parse as urlparse
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 크롬드라이버 초기 설정
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# 결과 누적 리스트
info_list = []

# 텍스트 파일에서 list URL 읽기
with open("webnovel_data/로맨스.txt", "r", encoding="utf-8") as f:
    list_urls = [line.strip() for line in f if line.strip()]

for list_url in list_urls:
    try:
        parsed = urlparse.urlparse(list_url)
        novel_id = urlparse.parse_qs(parsed.query)['novelId'][0]

        # 접속
        driver.get(list_url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 웹소설 기본 정보
        title_tag = soup.find('h2', class_='title')
        title = title_tag.text.strip() if title_tag else f"작품_{novel_id}"

        genre_tag = soup.find('span', class_='item')
        genre = genre_tag.text.strip() if genre_tag else '장르 없음'

        score_tag = soup.select_one('span.score_area')
        score = score_tag.get_text(strip=True).replace('별점', '') if score_tag else '별점 없음'

        download_tag = soup.select_one('span.count')
        download = download_tag.get_text(strip=True) if download_tag else '다운로드 수 없음'

        concern_tag = soup.find('span', id='concernCount')
        concern = concern_tag.text.strip() if concern_tag else '관심 수 없음'

        # 줄거리 수집
        summary_tag = soup.find('span', id='summaryText')
        summary = summary_tag.get_text(strip=True) if summary_tag else '줄거리 없음'

        # 정보 누적
        info_list.append({
            "제목": title,
            "장르": genre,
            "별점": score,
            "다운로드 수": download,
            "관심 수": concern,
            "줄거리": summary
        })

        print(f"처리 완료: {title}")

    except Exception as e:
        print(f"오류 발생: {list_url}")
        print(e)

# 드라이버 종료
driver.quit()

# CSV로 저장
df_info = pd.DataFrame(info_list)
df_info.to_csv("webnovel_data/로맨스_기본정보_줄거리포함.csv", index=False, encoding="utf-8-sig")
print("CSV 저장 완료 (줄거리 포함)")
