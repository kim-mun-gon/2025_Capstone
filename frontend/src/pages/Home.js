import React from "react";
import TagFilter from "../components/TagFilter";
import PredictButton from "../components/PredictButton";
import RankingCard from "../components/RankingCard";

function Home() {
  return (
    <div className="home">
      <TagFilter />

      <PredictButton />

      <div className="ranking-section">
      <RankingCard title="네이버 웹소설 랭킹" list={[]} />
      <RankingCard title="10대 랭킹" list={[]} />
      <RankingCard title="20대 랭킹" list={[]} />
      <RankingCard title="30대 랭킹" list={[]} />
      <RankingCard title="40대 랭킹" list={[]} />
      </div>
    </div>
  );
}

export default Home;
