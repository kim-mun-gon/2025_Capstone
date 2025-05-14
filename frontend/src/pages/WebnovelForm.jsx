import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./WebnovelForm.css";

const WebnovelForm = () => {
  const [title, setTitle]             = useState("");
  const [genre, setGenre]             = useState("");
  const [summary, setSummary]         = useState("");
  const [loading, setLoading]         = useState(false);
  const [progress, setProgress]       = useState(0);
  const [showDone, setShowDone]       = useState(false);
  const navigate                       = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setProgress(10);
    setShowDone(false);

    setTimeout(async () => {
      try {
        const [
          simRes,
          successRes,
          scoreRes,
          noveltyRes,
          impactRes
        ] = await Promise.all([
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
          fetch("http://localhost:5001/api/novelty-score", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, genre, summary }),
          }),
          fetch("http://localhost:5001/api/impact-score", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, genre, summary }),
          }),
        ]);

        if (
          !simRes.ok ||
          !successRes.ok ||
          !scoreRes.ok ||
          !noveltyRes.ok ||
          !impactRes.ok
        ) {
          throw new Error("API 중 하나 이상 실패");
        }

        const simResult     = await simRes.json();
        const successResult = await successRes.json();
        const scoreResult   = await scoreRes.json();
        const noveltyResult = await noveltyRes.json();
        const impactResult  = await impactRes.json();

        localStorage.setItem(
          "analysisResult",
          JSON.stringify({
            similar_novels:    simResult.similar_novels,
            similar_titles:    simResult.similar_titles,
            predicted_rating:  scoreResult.predicted_rating,
            positive_rate:     scoreResult.positive_rate,
            success_probability: successResult.success_probability,
            grade:             successResult.grade,
            feedback:          successResult.feedback,
            novelty_score:     noveltyResult.novelty_score,
            novelty_keywords:  noveltyResult.user_keywords,
            overlap_keywords:  noveltyResult.overlap_keywords,
            overlap_ratio:     noveltyResult.overlap_ratio,
            impact_score:      impactResult.impact_score,
            emotion_hits:      impactResult.emotion_hits,
            motif_hits:        impactResult.motif_hits,
            transition_hits:   impactResult.transition_hits,
          })
        );

        await new Promise((resolve) => setTimeout(resolve, 1000));
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
      setProgress((prev) => (prev >= 95 ? prev : prev + 1));
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
              />
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
              placeholder="장르를 입력하세요 (예: 로맨스, 판타지, 로판, 무협, 미스터리, 현판)"
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
