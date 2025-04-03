import React, { useState } from "react";
import axios from "axios";
import "./Predictor.css";

const Predictor = () => {
  const [inputs, setInputs] = useState({
    recipe_name: "",
    cooking_time: "",
    calories: ""
  });

  const [prediction, setPrediction] = useState(null);
  const [percentageRank, setPercentageRank] = useState(null);
  const [showRank, setShowRank] = useState(false); // State to control when to show rank
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setShowRank(false); // Hide rank before submitting

    try {
      const formattedInputs = {
        recipe_name: inputs.recipe_name,
        cooking_time: parseFloat(inputs.cooking_time) || 0,
        calories: parseFloat(inputs.calories) || 0
      };

      console.log("Sending Data:", formattedInputs);

      const response = await axios.post("http://127.0.0.1:5000/predict", formattedInputs, {
        headers: { "Content-Type": "application/json" }
      });

      console.log("Received Response:", response.data);

      if (response.data) {
        setPrediction(`${response.data.predicted_popularity} / 5`);
        setPercentageRank(response.data.top_percentage);
        setShowRank(true); // Show rank only after getting response
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
          {[{ name: "recipe_name", type: "text", label: "Recipe Name" },
            { name: "cooking_time", type: "number", label: "Cooking Time (mins)" },
            { name: "calories", type: "number", label: "Calories" }
          ].map((field) => (
            <div key={field.name} className="input-container">
              <label className="input-label">{field.label}:</label>
              <input
                type={field.type}
                name={field.name}
                value={inputs[field.name]}
                onChange={handleChange}
                required
                className="input-field"
              />
            </div>
          ))}
          <button type="submit" className="predictor-button">Predict</button>
        </form>
        {error && <p className="error-message">{error}</p>}
        {prediction && <p><strong>Score:</strong> {prediction}</p>}
        {showRank && percentageRank !== null && (
          <p>🏆 This recipe ranks in the <em>top {percentageRank.toFixed(1)}%</em> of all recipes.</p>
        )}
      </div>
    </div>
  );
};

export default Predictor;
