import React, { useEffect, useState } from "react";
import TagFilter from "../components/TagFilter";
import PredictButton from "../components/PredictButton";
import RankingCard from "../components/RankingCard";
import "./Home.css";

function Home() {
  const [naverRanking, setNaverRanking] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5001/api/naver-rankings")
      .then(res => res.json())
      .then(data => setNaverRanking(data));
  }, []);

  return (
    <div className="home">
      <TagFilter />
      <PredictButton />
      <div className="ranking-section">
        <RankingCard title="네이버 웹소설 랭킹" list={naverRanking} />
        <RankingCard title="10대 랭킹" list={[]} />
        <RankingCard title="20대 랭킹" list={[]} />
        <RankingCard title="30대 랭킹" list={[]} />
        <RankingCard title="40대 랭킹" list={[]} />
      </div>
    </div>
  );
}

export default Home;
