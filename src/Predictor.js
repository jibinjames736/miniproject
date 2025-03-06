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
  
      // Ensure that the response has the 'predicted_popularity_score' field
      if (response.data && response.data.predicted_popularity_score !== undefined) {
        setPrediction(`${response.data.predicted_popularity_score} / 1000`); // Append "/ 1000"
      } else {
        setError("Error: 'predicted_popularity_score' not found in response.");
      }
    } catch (error) {
      console.error("Error:", error.response?.data || error.message);
      setError("Failed to fetch prediction. Please try again.");
    }
  };

  return (
    <div className="predictor-container">
      <div className="predictor-box">
        <h2 className="predictor-heading">Popularity Score</h2> {/* Heading */}
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
            <p>{prediction}</p> {/* Display the score with "/1000" */}
          </div>
        )}
      </div>
    </div>
  );
};

export default Predictor;
