import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor

# Load dataset
file_path = "reviews.csv"
df = pd.read_csv(file_path)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Drop unnecessary columns
columns_to_drop = ["Unnamed: 0", "comment_id", "user_id", "user_name", "text", "recipe_number", "recipe_code", "created_at"]
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Drop missing values
df_cleaned = df_cleaned.dropna()

# Ensure 'best_score' is present
if "best_score" not in df_cleaned.columns:
    raise KeyError("Column 'best_score' is missing from the dataset. Check column names!")

# Encode 'recipe_name' if available
if "recipe_name" in df_cleaned.columns:
    encoder = LabelEncoder()
    df_cleaned["recipe_name_encoded"] = encoder.fit_transform(df_cleaned["recipe_name"])

# Define features (X) and target variable (y)
feature_names = ["user_reputation", "reply_count", "thumbs_up", "thumbs_down", "stars", "recipe_name_encoded", "calories", "cooking_time"]
X = df_cleaned[feature_names]
y_cleaned = df_cleaned["best_score"].values.reshape(-1, 1)

# Scale y (popularity score) between 0 and 5
y_scaler = MinMaxScaler(feature_range=(0, 5))
y_scaled = y_scaler.fit_transform(y_cleaned).flatten()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model performance
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R-squared value of the model: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Save model and scalers
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")

# Load saved model and scalers
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict_popularity():
    try:
        data = request.json
        if not data or "recipe_name" not in data:
            return jsonify({"error": "Missing 'recipe_name' in request"}), 400

        recipe_name = data["recipe_name"]
        target_calories = data.get("calories")
        target_cooking_time = data.get("cooking_time")

        if recipe_name not in encoder.classes_:
            return jsonify({"error": "Recipe not found in dataset"}), 404

        recipe_subset = df_cleaned[df_cleaned["recipe_name"] == recipe_name].copy()
        if recipe_subset.empty:
            return jsonify({"error": "No records found for given recipe name"}), 404

        # Handle missing target values
        sort_columns = []
        if target_calories is not None:
            recipe_subset["calories_diff"] = abs(recipe_subset["calories"] - target_calories)
            sort_columns.append("calories_diff")
        if target_cooking_time is not None:
            recipe_subset["cooking_time_diff"] = abs(recipe_subset["cooking_time"] - target_cooking_time)
            sort_columns.append("cooking_time_diff")

        if sort_columns:
            recipe_subset = recipe_subset.sort_values(by=sort_columns, ascending=True)

        selected_record = recipe_subset.iloc[0][feature_names].copy()
        input_df = pd.DataFrame([selected_record])
        input_scaled = scaler.transform(input_df)

        # Get prediction directly in the 0-5 range
        prediction_scaled = model.predict(input_scaled)
        prediction = np.clip(prediction_scaled, 0, 5)[0]

        return jsonify({
            "predicted_popularity": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
