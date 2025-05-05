import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import WebnovelForm from "./pages/WebnovelForm";
import Result from "./pages/Result"; // ✅ 결과 페이지 import 추가

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/webnovel-form" element={<WebnovelForm />} />
        <Route path="/result" element={<Result />} /> {/* ✅ 결과 페이지 라우팅 추가 */}
      </Routes>
    </Router>
  );
}

export default App;
