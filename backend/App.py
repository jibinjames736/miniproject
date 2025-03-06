from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Load the dataset
file_path = "reviews.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns (adjust based on dataset structure)
columns_to_drop = ["Unnamed: 0", "comment_id", "user_id", "user_name", "text", 
                   "recipe_name", "recipe_number", "recipe_code", "created_at"]

df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Drop rows with missing values
df_cleaned = df_cleaned.dropna()

# Define features (X) and target variable (y)
X = df_cleaned.drop(columns=["best_score"])
y = df_cleaned["best_score"]

# Remove outliers using Z-score (values beyond 3 standard deviations)
z_scores = np.abs((X - X.mean()) / X.std())
X_cleaned = X[(z_scores < 3).all(axis=1)]
y_cleaned = y.loc[X_cleaned.index]  # Ensure corresponding y values are kept

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cleaned)

# Split the cleaned and scaled data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cleaned, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and scaler for later use
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load model and scaler when the server starts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict_popularity():
    try:
        data = request.json  # Receive JSON data
        print("Received Data:", data)  # Print received data
        
        # Ensure all necessary columns are provided in the input
        input_features = {feature: data.get(feature) for feature in X.columns}
        
        # Check if any feature is missing
        if None in input_features.values():
            return jsonify({"error": "Missing required feature values in input data"}), 400
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_features], columns=X.columns)  
        input_scaled = scaler.transform(input_df)  # Scale input
        prediction = model.predict(input_scaled)  # Predict
        
        return jsonify({"predicted_popularity_score": round(prediction[0], 2)})  # Return prediction as JSON

    except Exception as e:
        print("Error:", str(e))  # Print error details
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
