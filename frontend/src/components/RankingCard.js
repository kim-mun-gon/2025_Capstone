import React from "react";

const RankingCard = ({ title, list = [] }) => (
  <div className="ranking-card">
    <h4 style={{ marginBottom: "12px" }}>{title}</h4>
    <ul style={{ listStyle: "none", padding: 0 }}>
      {[...Array(5)].map((_, idx) => (
        <li
          key={idx}
          style={{
            display: "flex",
            marginBottom: "12px",
            alignItems: "center",
          }}
        >
          <div
            style={{
              width: "50px",
              height: "70px",
              backgroundColor: "#eee",
              borderRadius: "4px",
              marginRight: "10px",
            }}
          />
          <div>
            <div
              style={{
                width: "120px",
                height: "14px",
                backgroundColor: "#ddd",
                marginBottom: "6px",
              }}
            />
            <div
              style={{
                width: "80px",
                height: "12px",
                backgroundColor: "#eee",
              }}
            />
          </div>
        </li>
      ))}
    </ul>
  </div>
);

export default RankingCard;
