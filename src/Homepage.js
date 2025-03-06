import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Homepage.css';

export default function HomePage() {
    const navigate = useNavigate();

    return (
        <div className="homepage-container">
            <h1 className="homepage-title">
                Recipe Popularity Prediction System
            </h1>
            
            <p className="homepage-description">
                Discover which recipes are trending and analyze their popularity with our AI-powered system.
            </p>
            
            <div className="button-container">
                <button 
                    onClick={() => navigate('/predictor')} // Fixed route path (all lowercase)
                    className="homepage-button popularity-button"
                >
                    Popularity Score
                </button>
                
                <button 
                    onClick={() => navigate('/trend-analyzer')}
                    className="homepage-button trend-button"
                >
                    Trend Analyzer
                </button>
            </div>
        </div>
    );
}
