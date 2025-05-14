import React, { useEffect, useState } from "react";
import "./TagFilter.css";

const TagFilter = () => {
  const [tags, setTags] = useState([]);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5001/api/tags")
      .then((res) => res.json())
      .then((data) => {
        setTags(data);
        setSelected(data[0]);
      });
  }, []);

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
