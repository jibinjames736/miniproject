import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./Homepage";
import Predictor from "./Predictor"; // Import Predictor component
import "./App.css";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/Predictor" element={<Predictor />} /> {/* Add Predictor route */}
      </Routes>
    </Router>
  );
}

export default App;