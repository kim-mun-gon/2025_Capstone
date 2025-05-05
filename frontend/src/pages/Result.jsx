import React, { useState, useEffect } from "react";
import "./Result.css";

const Result = () => {
  const [activeTab, setActiveTab] = useState("similar");
  const [similarResults, setSimilarResults] = useState([]);
  const [predictedRating, setPredictedRating] = useState(null);
  const [positiveRate, setPositiveRate] = useState(null);
  const [success, setSuccess] = useState(null);
  const [grade, setGrade] = useState("");
  const [feedback, setFeedback] = useState([]);

  useEffect(() => {
    const stored = localStorage.getItem("analysisResult");
    if (!stored) return;
    const parsed = JSON.parse(stored);

    if (parsed.similar_novels) {
      setSimilarResults(parsed.similar_novels);
    }
    if (parsed.predicted_rating != null) {
      setPredictedRating(parsed.predicted_rating);
    }
    if (parsed.positive_rate != null) {
      setPositiveRate(parsed.positive_rate * 100); // 0~1 to %
    }
    if (parsed.success_probability != null) {
      setSuccess(Math.min(parsed.success_probability * 10, 100));
    }
    if (parsed.grade) {
      setGrade(parsed.grade);
    }
    if (Array.isArray(parsed.feedback)) {
      setFeedback(parsed.feedback);
    }
  }, []);

  return (
    <div className="result-container">
      <div className="card-row">
        <div className="result-card">
          <div className="title">Predicted Rating</div>
          <div className="value">
            {predictedRating != null ? predictedRating.toFixed(1) : "-"}
            <small>/10</small>
          </div>
        </div>

        <div className="result-card">
          <div className="title">Success Possibility</div>
          <div className="value">
            {success != null ? success.toFixed(0) : "-"}%
            {grade && <span className={`badge grade-${grade}`}>{grade}</span>}
          </div>
        </div>

        <div className="result-card">
          <div className="title">Positive Comment Rate</div>
          <div className="value">
            {positiveRate != null ? positiveRate.toFixed(0) : "-"}%
          </div>
        </div>
      </div>

      <div className="tab-buttons">
        <button
          className={activeTab === "similar" ? "active" : ""}
          onClick={() => setActiveTab("similar")}
        >
          Similar Novels
        </button>
        <button
          className={activeTab === "genre" ? "active" : ""}
          onClick={() => setActiveTab("genre")}
        >
          Genre Analysis
        </button>
      </div>

      {activeTab === "similar" && (
        <div className="similar-novels">
          <h3>Similar Successful Novels</h3>
          <ul>
            {similarResults.length > 0 ? (
              similarResults.map((item, idx) => (
                <li key={idx}>
                  <span className="rank">#{idx + 1}</span>
                  <strong>{item.title}</strong>
                  <span className="match">{item.similarity} match</span>
                </li>
              ))
            ) : (
              <li>⏳ 분석된 유사 작품이 없습니다.</li>
            )}
          </ul>
        </div>
      )}

      {activeTab === "genre" && (
        <div className="genre-analysis">
          <h3>Genre & Trend Analysis</h3>
          <p className="sub">How your novel aligns with current trends</p>
        </div>
      )}

      <div className="suggestions">
        <h3>Improvement Suggestions</h3>
        <ul>
          {feedback.length > 0 ? (
            feedback.map((item, idx) => <li key={idx}>{item}</li>)
          ) : (
            <li>No suggestions available.</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default Result;
