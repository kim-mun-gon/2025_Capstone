import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import WebnovelForm from "./pages/WebnovelForm";  // ✅ .jsx여도 import는 .js처럼 생략 가능

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/webnovel-form" element={<WebnovelForm />} /> {/* ✅ 이 경로 추가 */}
      </Routes>
    </Router>
  );
}

export default App;
