from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from flask_cors import CORS  # Import CORS

# Load dataset
file_path = "reviews.csv"
df = pd.read_csv(file_path)

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Drop unnecessary columns
columns_to_drop = ["Unnamed: 0", "comment_id", "user_id", "user_name", "text", 
                   "recipe_name", "recipe_number", "recipe_code", "created_at"]
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Drop rows with missing values
df_cleaned = df_cleaned.dropna()

# Define features (X) and target variable (y)
X = df_cleaned.drop(columns=["best_score"])
y = df_cleaned["best_score"]

# Remove outliers using Z-score
z_scores = np.abs((X - X.mean()) / X.std())
X_cleaned = X[(z_scores < 3).all(axis=1)]
y_cleaned = y.loc[X_cleaned.index]  

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cleaned)

# Scale target variable (best_score) to range 0-5
y_scaler = MinMaxScaler(feature_range=(0, 5))
y_scaled = y_scaler.fit_transform(y_cleaned.values.reshape(-1, 1)).ravel()  # Keep it 1D

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scalers
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

print("✅ Model training complete. Saved model and scalers.")

# Calculate accuracy
accuracy = model.score(X_test, y_test) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# Load model and scalers
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# Get feature names
X_columns = X_cleaned.columns.tolist()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Ensure all required features are provided
        missing_features = [feature for feature in X_columns if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Predict score
        prediction = model.predict(input_scaled)
        
        # Scale prediction back to 0-5 range
        prediction_scaled = np.clip(prediction, 0, 5)
        
        return jsonify({"predicted_popularity_score": round(prediction_scaled[0], 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
