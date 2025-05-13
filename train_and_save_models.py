import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

print("Training and saving ML models for Y_BoxOffice predictions...")

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load the data
try:
    df_model = pd.read_csv('TMDB_cleaned_data.csv')
    print(f"Data loaded successfully. Shape: {df_model.shape}")
except FileNotFoundError:
    print("Error: TMDB_cleaned_data.csv not found. Please run the EDA notebook first.")
    exit(1)

# 2. Select features and target (following notebook logic)
features = ['runtime', 'vote_average', 'vote_count', 'release_year', 'release_month']
target = 'log_revenue'

# 3. Prepare data
df_model = df_model[features + [target]].dropna()
X = df_model[features]
y = df_model[target]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 5. Define and train models
models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=15,
        min_samples_leaf=5
    )
}

results = {}

# 6. Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    results[name] = {'model': model, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    # Save model
    with open(f"models/{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved as models/{name}_model.pkl")

# 7. Save feature importance for Random Forest
if 'random_forest' in results:
    rf_model = results['random_forest']['model']
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    print("Feature importance saved to models/feature_importance.csv")

print("\nModel training and saving complete!")
print(f"Best model: {max(results.items(), key=lambda x: x[1]['R2'])[0]}") 