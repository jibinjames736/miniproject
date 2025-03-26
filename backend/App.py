import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load dataset
file_path = "reviews.csv"
df = pd.read_csv(file_path)
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# Drop unnecessary columns
columns_to_drop = ["Unnamed: 0", "comment_id", "user_id", "user_name", "text", 
                   "recipe_name", "recipe_number", "recipe_code", "created_at"]
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Drop missing values
df_cleaned = df_cleaned.dropna()

# Ensure 'best_score' is present
if "best_score" not in df_cleaned.columns:
    raise KeyError("Column 'best_score' is missing from the dataset. Check column names!")

# Define features (X) and target variable (y)
X = df_cleaned.drop(columns=["best_score"])
y_cleaned = df_cleaned["best_score"]

# Save the original popularity scores before scaling
joblib.dump(y_cleaned, "popularity_data.pkl")

# Scale y (popularity score) between 0 and 5
y_scaler = MinMaxScaler(feature_range=(0, 5))
y_scaled = y_scaler.fit_transform(y_cleaned.values.reshape(-1, 1)).flatten()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on test set
y_pred_scaled = xgb_model.predict(X_test)

# Ensure predictions stay within 0-5 range
y_pred = np.clip(y_pred_scaled, 0, 5)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model and scalers
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

# Display results
print("\n✅ Model training complete! All files saved.")
print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📊 R² Score: {r2:.4f}")

# Load the saved model and scalers


with app.app_context():  # Ensures the app context is properly set
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")  # Feature scaler
    y_scaler = joblib.load("y_scaler.pkl")  # Target scaler
    popularity_data = joblib.load("popularity_data.pkl")  # Original popularity scores


# Feature names used in training
feature_names = ["user_reputation", "reply_count", "thumbs_up", "thumbs_down", "stars"]


@app.route("/predict", methods=["POST"])
def predict_popularity():

  try:
    """Predicts popularity score & displays insights."""
    data = request.json
    if not data:
            
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to DataFrame
    input_df = pd.DataFrame([data])
    
    # Ensure input uses the same feature order as training
    data = scaler.transform(input_df)  # Scale input features

    # Predict scaled popularity score
    prediction_scaled = model.predict(data)

    # Convert back to original scale
    prediction_scaled = np.array(prediction_scaled).reshape(-1, 1)  # Fix reshaping issue
    prediction = y_scaler.inverse_transform(prediction_scaled)[0][0]  # Get actual value

    # Scale the predicted popularity score to a 0-5 range
    min_score, max_score = 0, 5
    normalized_score = np.clip((prediction - popularity_data.min()) / (popularity_data.max() - popularity_data.min()) * 5, min_score, max_score)

    # Get insights from data
    surpassing_count = (popularity_data >= 4).sum()
    percentage_rank = 100 - ((popularity_data > prediction).mean() * 100)

    # Display results
    print("\n🎯 *Predicted Popularity Score (0-5): {:.2f}*".format(normalized_score))
    print("📈 {} recipes ({:.2f}%) have a popularity score ≥ 4.".format(surpassing_count, (surpassing_count / len(popularity_data)) * 100))
    print("🏆 This recipe ranks in the *top {:.1f}%* of all recipes.".format(percentage_rank))

    return jsonify({
            "predicted_popularity": round(normalized_score, 2),
            "surpassing_count": int(surpassing_count),
            "percentage_rank": round(percentage_rank, 1),
            "popularity_data":int(len(popularity_data))
        })
  except Exception as e:
      
        return jsonify({"error": str(e)}), 500

# Run prediction
if __name__ == "__main__":
    app.run(debug=True)
