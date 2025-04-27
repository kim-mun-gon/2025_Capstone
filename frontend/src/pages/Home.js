import React, { useEffect, useState } from "react";
import TagFilter from "../components/TagFilter";
import PredictButton from "../components/PredictButton";
import RankingCard from "../components/RankingCard";
import "./Home.css";

function Home() {
  const [naverRanking, setNaverRanking] = useState([]);
  const [teenRanking, setTeenRanking] = useState([]);
  const [twentyRanking, setTwentyRanking] = useState([]);
  const [thirtyRanking, setThirtyRanking] = useState([]);
  const [fortyRanking, setFortyRanking] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5001/api/naver-rankings")
      .then(res => res.json())
      .then(data => setNaverRanking(data));

    fetch("http://localhost:5001/api/rankings/teen")
      .then(res => res.json())
      .then(data => setTeenRanking(data));

    fetch("http://localhost:5001/api/rankings/twenty")
      .then(res => res.json())
      .then(data => setTwentyRanking(data));

    fetch("http://localhost:5001/api/rankings/thirty")
      .then(res => res.json())
      .then(data => setThirtyRanking(data));

    fetch("http://localhost:5001/api/rankings/forty")
      .then(res => res.json())
      .then(data => setFortyRanking(data));
  }, []);

  return (
    <div className="home">
      <TagFilter />
      <PredictButton />
      <div className="ranking-section">
        {naverRanking.length === 0 ? <div className="loading">로딩 중...</div> : <RankingCard title="네이버 웹소설 랭킹" list={naverRanking} />}
        {teenRanking.length === 0 ? <div className="loading">로딩 중...</div> : <RankingCard title="10대 랭킹" list={teenRanking} />}
        {twentyRanking.length === 0 ? <div className="loading">로딩 중...</div> : <RankingCard title="20대 랭킹" list={twentyRanking} />}
        {thirtyRanking.length === 0 ? <div className="loading">로딩 중...</div> : <RankingCard title="30대 랭킹" list={thirtyRanking} />}
        {fortyRanking.length === 0 ? <div className="loading">로딩 중...</div> : <RankingCard title="40대 랭킹" list={fortyRanking} />}
      </div>
    </div>
  );
}

export default Home;
