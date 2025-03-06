import React, { useState } from 'react';
import './PScore.css';

export default function PScore() {
    const [inputs, setInputs] = useState({
        ingredientCount: '',
        cookTime: '',
        calories: '',
        userRatings: '',
        reviewCount: '',
        spiceLevel: ''
    });
    
    const handleChange = (e) => {
        const { name, value } = e.target;
        setInputs(prev => ({ ...prev, [name]: value }));
    };

    const handleAnalyze = () => {
        alert(`Predicted Popularity Score based on input data!`);
    };

    return (
        <div className="pscore-container">
            <h1 className="pscore-title">Popularity Score Predictor</h1>
            <p className="pscore-description">Enter the recipe details below to predict its popularity score.</p>
            
            <div className="pscore-form">
                <input type="text" name="ingredientCount" placeholder="Number of Ingredients" value={inputs.ingredientCount} onChange={handleChange} />
                <input type="text" name="cookTime" placeholder="Cooking Time (mins)" value={inputs.cookTime} onChange={handleChange} />
                <input type="text" name="calories" placeholder="Calories" value={inputs.calories} onChange={handleChange} />
                <input type="text" name="userRatings" placeholder="User Ratings (out of 5)" value={inputs.userRatings} onChange={handleChange} />
                <input type="text" name="reviewCount" placeholder="Number of Reviews" value={inputs.reviewCount} onChange={handleChange} />
                <input type="text" name="spiceLevel" placeholder="Spice Level (1-10)" value={inputs.spiceLevel} onChange={handleChange} />
                
                <button className="analyze-button" onClick={handleAnalyze}>Analyze</button>
            </div>
        </div>
    );
}
