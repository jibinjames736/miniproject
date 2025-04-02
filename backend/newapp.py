import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = input("Enter the path to your dataset: ")
df = pd.read_csv(file_path)

# User selects relevant columns
features = input("Enter the feature columns (comma-separated): ").split(',')
target = input("Enter the target column: ")

# Normalize target variable
target_range = (0, 5)
scaler = MinMaxScaler(feature_range=target_range)
df['target_normalized'] = scaler.fit_transform(df[[target]])

# Encode categorical features
for col in features:
    if df[col].dtype == 'object':
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

# Prepare data for training
X = df[features]
y = df['target_normalized']

# Split into training and test sets
test_size = float(input("Enter test size (e.g., 0.2 for 20%): "))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train a Random Forest Regressor model
n_estimators = int(input("Enter the number of estimators for RandomForestRegressor: "))
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
