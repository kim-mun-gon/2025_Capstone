import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./WebnovelForm.css";

const WebnovelForm = () => {
  const [title, setTitle] = useState("");
  const [genre, setGenre] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showDone, setShowDone] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setProgress(10); // 처음에 10부터 시작
    setShowDone(false);

    setTimeout(async () => {
      try {
        const [simRes, successRes, scoreRes] = await Promise.all([
          fetch("http://localhost:5001/api/similarity", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, genre, summary }),
          }),
          fetch("http://localhost:5001/api/success-predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, genre, summary }),
          }),
          fetch("http://localhost:5001/api/predict-score", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, genre, summary }),
          }),
        ]);

        if (!simRes.ok || !successRes.ok || !scoreRes.ok) {
          throw new Error("API 중 하나 이상 실패");
        }

        const simResult = await simRes.json();
        const successResult = await successRes.json();
        const scoreResult = await scoreRes.json();

        localStorage.setItem(
          "analysisResult",
          JSON.stringify({
            similar_novels: simResult.similar_novels,
            similar_titles: simResult.similar_titles,
            predicted_rating: scoreResult.predicted_rating,
            positive_rate: scoreResult.positive_rate,
            success_probability: successResult.success_probability,
            grade: successResult.grade,
            feedback: successResult.feedback,
          })
        );

        await new Promise((resolve) => setTimeout(resolve, 1000)); // 최소 1초 대기
        setProgress(100);
        setShowDone(true);
        setTimeout(() => navigate("/result"), 1000);
      } catch (error) {
        console.error("❌ 분석 오류:", error);
        alert("분석 중 오류가 발생했습니다.");
        setLoading(false);
      }
    }, 0);
  };

  useEffect(() => {
    if (!loading) return;

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) return prev;
        return prev + 1;
      });
    }, 30);

    return () => clearInterval(interval);
  }, [loading]);

  return (
    <div className="webnovel-form-wrapper">
      {!loading && <h2 className="form-title">📖 웹소설 정보 입력</h2>}

      {loading ? (
        <div className="loading-section">
          <h2 className="loading-text">
            {showDone
              ? "✅ 분석 완료! 결과 페이지로 이동 중..."
              : `✨ 작품 분석 중입니다... ${progress}%`}
          </h2>

          {!showDone && (
            <div className="progress-bar-wrapper">
              <div
                className="progress-bar"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          )}
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="input-form">
          <div>
            <label>제목</label>
            <input
              type="text"
              placeholder="웹소설 제목을 입력하세요"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
            />
          </div>

          <div>
            <label>장르</label>
            <input
              type="text"
              placeholder="장르를 입력하세요 (예: 로맨스, 판타지)"
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
              required
            />
          </div>

          <div>
            <label>줄거리</label>
            <textarea
              placeholder="줄거리를 작성해주세요"
              value={summary}
              onChange={(e) => setSummary(e.target.value)}
              required
              rows="6"
            />
          </div>

          <button type="submit" className="submit-btn">
            제출하기
          </button>
        </form>
      )}
    </div>
  );
};

export default WebnovelForm;
