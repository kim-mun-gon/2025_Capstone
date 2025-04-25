import React from "react";

const RankingCard = ({ title, list = [] }) => (
  <div
    className="ranking-card"
    style={{
      width: "240px",
      minWidth: "240px",
      borderRadius: "10px",
      padding: "16px",
      backgroundColor: "#fff",
      boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
    }}
  >
    <h4
      style={{
        marginBottom: "12px",
        fontSize: "16px",
        fontWeight: "bold",
        textAlign: "left",
      }}
    >
      {title}
    </h4>

    <ul
      style={{
        listStyle: "none",
        padding: 0,
        margin: 0,
        maxHeight: "400px",
        overflowY: "auto",
        scrollbarWidth: "thin",
      }}
    >
      {list.map((item, idx) => {
        const cleanTitle = item.title.replace(/^(UP|NEW|신작)\s+/i, "");

        return (
          <li
            key={idx}
            style={{
              display: "flex",
              alignItems: "flex-start",
              marginBottom: "14px",
            }}
          >
            <a href={item.link} target="_blank" rel="noreferrer">
              <img
                src={item.image}
                alt={item.title}
                style={{
                  width: "50px",
                  height: "70px",
                  objectFit: "cover",
                  borderRadius: "6px",
                  marginRight: "10px",
                }}
              />
            </a>

            <div
              style={{
                fontSize: "13px",
                display: "flex",
                flexDirection: "column",
                justifyContent: "flex-start",
                alignItems: "flex-start",
                width: "170px",
                textAlign: "left",
              }}
            >
              <div
                style={{
                  fontWeight: "bold",
                  marginBottom: "4px",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  width: "170px",
                }}
              >
                {idx + 1}. {cleanTitle}
              </div>

              <div
                style={{
                  color: "#555",
                  fontSize: "12px",
                  marginBottom: "2px",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  width: "170px",
                }}
              >
                {item.author} · {item.genre}
              </div>

              <div
                style={{
                  color: "#aaa",
                  fontSize: "12px",
                  fontVariantNumeric: "tabular-nums",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  width: "170px",
                }}
              >
                 {item.count}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  </div>
);

export default RankingCard;
