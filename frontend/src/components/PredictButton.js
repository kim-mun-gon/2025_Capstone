import { useNavigate } from "react-router-dom";
import "./PredictButton.css";

const PredictButton = () => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate("/webnovel-form");  // ✅ 여기 정확히 '/webnovel-form'로 이동
  };

  return (
    <div className="predict-wrapper">
      <button className="predict-btn" onClick={handleClick}>
        당신의 웹소설의 예상되는 평가가 궁금하다면?
      </button>
    </div>
  );
};

export default PredictButton;
