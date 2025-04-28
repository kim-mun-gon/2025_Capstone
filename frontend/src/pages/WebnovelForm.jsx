import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";  // ✅ 결과 페이지 이동용
import "./WebnovelForm.css"; // ✅ 별도 CSS 파일 불러오기

const WebnovelForm = () => {
  const [title, setTitle] = useState("");
  const [genre, setGenre] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const navigate = useNavigate(); // ✅ 이동용

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setProgress(0);
  };

  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setTimeout(() => navigate("/result"), 1000); // ✅ 1초 후 결과페이지로 이동
            return 100;
          }
          return prev + 1;
        });
      }, 30);
    }
  }, [loading, navigate]);

  return (
    <div className="webnovel-form-wrapper">
      {/* 로딩 아닐 때만 타이틀 보여주기 */}
      {!loading && (
        <h2 className="form-title">📖 웹소설 정보 입력</h2>
      )}

      {/* 로딩 중 */}
      {loading ? (
        <div className="loading-section">
          <h2 className="loading-text">
            ✨ 사용자님의 작품에 대한 분석 중입니다. 잠시만 기다려주세요. {progress}%
          </h2>
          <div className="progress-bar-wrapper">
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          </div>
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

          <button type="submit" className="submit-btn">제출하기</button>
        </form>
      )}
    </div>
  );
};

export default WebnovelForm;
