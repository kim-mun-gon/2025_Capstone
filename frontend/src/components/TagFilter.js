import React, { useState } from "react";
import "./TagFilter.css";

const TagFilter = () => {
  const tags = ["#재벌님", "#순정남", "#술술읽히는", "#다정남", "#계약관계"];
  const [selected, setSelected] = useState("#재벌님");

  return (
    <div className="tag-wrapper">
      <h3 className="tag-title">현재 인기있는 소설 태그</h3>
      <div className="tag-container">
        {tags.map((tag) => (
          <button
            key={tag}
            className={`tag-button ${selected === tag ? "selected" : ""}`}
            onClick={() => setSelected(tag)}
          >
            {tag}
          </button>
        ))}
      </div>
    </div>
  );
};

export default TagFilter;
