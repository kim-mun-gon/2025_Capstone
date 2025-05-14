## 🛠️ 필수  패키지 설치

아래 명령어로 모든 의존 패키지를 한 번에 설치할 수 있습니다: <br>

pip install pandas numpy tensorflow scikit-learn konlpy selenium beautifulsoup4 urllib3 webdriver-manager

|패키지 | 설명|
|------|---|
pandas, numpy | 데이터 처리용 기본 패키지
tensorflow | 감정 분석 모델 학습 및 예측
scikit-learn | ML 회귀 모델 및 정규화 처리
konlpy | 한국어 형태소 분석 (Okt 형태소 분석기 사용)
selenium, webdriver-manager | 웹 자동화 및 크롤링에 사용
beautifulsoup4 | HTML 파싱 및 텍스트 추출
urllib3 | URL 처리 및 웹 요청 처리용 라이브러리

🔎 urllib은 파이썬 내장 모듈이므로 별도 설치는 필요하지 않습니다.


## 📁 모델 및 데이터 파일

해당 프로젝트에서 사용된 모델 파일(`sentiment_model.h5`, `ml_rating_*.pkl`) 및  
학습/예측용 원본 데이터(`웹소설/*.csv`, `webnovel_data/*`)는  
파일 용량 제한(GitHub 100MB 제한)으로 인해 레포지토리에 포함되어 있지 않습니다.

### 🧠 직접 사용하려면?
- `sentiment_model.h5`: 감정 분석용 사전 학습 모델
- `웹소설/*.csv`: 각 장르별 유사 작품 정보
- `webnovel_data/*.csv`: 전체 원본 데이터
- `ml_rating_*.pkl`, `scaler_*.pkl`: 장르별 평점 예측용 모델 및 정규화 스케일러
