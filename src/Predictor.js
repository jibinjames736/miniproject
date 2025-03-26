import React, { useState } from "react";
import axios from "axios";
import "./Predictor.css"; // Import the CSS file

const Predictor = () => {
  const [inputs, setInputs] = useState({
    user_reputation: "",
    reply_count: "",
    thumbs_up: "",
    thumbs_down: "",
    stars: ""
  });

  const [prediction, setPrediction] = useState(null);
  const [surpassingCount, setSurpassingCount] = useState(null);
  const [percentageRank, setPercentageRank] = useState(null);
  const [popularityData, setPopularityData] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const formattedInputs = {
        user_reputation: parseFloat(inputs.user_reputation) || 0,
        reply_count: parseFloat(inputs.reply_count) || 0,
        thumbs_up: parseFloat(inputs.thumbs_up) || 0,
        thumbs_down: parseFloat(inputs.thumbs_down) || 0,
        stars: parseFloat(inputs.stars) || 0
      };

      console.log("Sending Data:", formattedInputs);

      const response = await axios.post("http://127.0.0.1:5000/predict", formattedInputs, {
        headers: { "Content-Type": "application/json" }
      });

      console.log("Received Response:", response.data);

      if (response.data) {
        setPrediction(`${response.data.predicted_popularity} / 5`);
        setSurpassingCount(response.data.surpassing_count);
        setPercentageRank(response.data.percentage_rank);
        setPopularityData(response.data.popularity_data);
      } else {
        setError("Error: Required fields not found in response.");
      }
    } catch (error) {
      console.error("Error:", error.response?.data || error.message);
      setError("Failed to fetch prediction. Please try again.");
    }
  };

  return (
    <div className="predictor-container">
      <div className="predictor-box">
        <h2 className="predictor-heading">Popularity Score</h2> 
        <form onSubmit={handleSubmit} className="predictor-form">
          {["user_reputation", "reply_count", "thumbs_up", "thumbs_down", "stars"].map((feature) => (
            <div key={feature} className="input-container">
              <label className="input-label">{feature.replace("_", " ")}:</label>
              <input
                type="number"
                name={feature}
                value={inputs[feature]}
                onChange={handleChange}
                required
                step="any"
                className="input-field"
              />
            </div>
          ))}
          <button type="submit" className="predictor-button">
            Predict
          </button>
        </form>
        {error && <p className="error-message">{error}</p>}
        {prediction && (
          <div className="prediction-result">
            <p><strong>Score:</strong> {prediction}</p>
            {/* <p>📈 {surpassingCount} recipes ({((surpassingCount / popularityData) * 100).toFixed(2)}%) have a popularity score ≥ 4.</p> */}
            <p>🏆 This recipe ranks in the <em>top {percentageRank?.toFixed(1)}%</em> of all recipes.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Predictor;
