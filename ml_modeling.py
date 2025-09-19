#!/usr/bin/env python3
"""
Machine Learning Modeling - Life Expectancy Prediction
Person 4 - ML Modeling

This script implements machine learning models to predict life expectancy.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=== MACHINE LEARNING MODELING - LIFE EXPECTANCY PREDICTION ===")
print("Person 4 - ML Modeling")
print()

# 1. Load and Explore Clean Dataset
print("1. Loading and exploring clean dataset...")
df = pd.read_csv('data/clean_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")
print()

# Basic dataset information
print("Dataset Info:")
print(df.info())
print()

print("Missing values:")
print(df.isnull().sum())
print()

print("Basic statistics:")
print(df.describe())
print()

# 2. Data Preparation for Modeling
print("2. Preparing data for modeling...")
target = 'life_expectancy'

# Check target variable distribution
print(f"Target variable statistics:")
print(df[target].describe())
print()

# Prepare features and target
exclude_cols = ['country', 'year', 'status', target]
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Features to use: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")
print()

# Create feature matrix and target vector
X = df[feature_cols]
y = df[target]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print()

# Check for any remaining missing values in features
missing_values = X.isnull().sum()
print("Missing values in features:")
print(missing_values[missing_values > 0])
print()

# If there are missing values, fill them
if missing_values.sum() > 0:
    print("Filling missing values with median...")
    X = X.fillna(X.median())
    print("Missing values after filling:", X.isnull().sum().sum())
else:
    print("No missing values found!")
print()

# 3. Train-Test Split
print("3. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training target mean: {y_train.mean():.2f}")
print(f"Test target mean: {y_test.mean():.2f}")
print()

# 4. Initial Model Training
print("4. Training initial models...")
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
}

initial_results = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Store results
    initial_results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting': abs(train_r2 - test_r2)
    }
    
    print(f"  Train RMSE: {train_rmse:.3f}")
    print(f"  Test RMSE: {test_rmse:.3f}")
    print(f"  Train MAE: {train_mae:.3f}")
    print(f"  Test MAE: {test_mae:.3f}")
    print(f"  Train R²: {train_r2:.3f}")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Overfitting (R² diff): {abs(train_r2 - test_r2):.3f}")
    print()

# 5. Model Comparison
print("5. Model comparison:")
comparison_data = []
for name, results in initial_results.items():
    comparison_data.append({
        'Model': name,
        'Train RMSE': results['train_rmse'],
        'Test RMSE': results['test_rmse'],
        'Train MAE': results['train_mae'],
        'Test MAE': results['test_mae'],
        'Train R²': results['train_r2'],
        'Test R²': results['test_r2'],
        'Overfitting': results['overfitting']
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.round(3))
print()

# 6. Cross-Validation
print("6. Performing cross-validation...")
cv_results = {}

for name, model in models.items():
    print(f"Cross-validation for {name}...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(cv_rmse_scores)
    
    cv_results[name] = {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std()
    }
    
    print(f"  CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"  CV RMSE: {cv_rmse_scores.mean():.3f} (+/- {cv_rmse_scores.std() * 2:.3f})")
    print()

# 7. Hyperparameter Optimization
print("7. Optimizing Random Forest hyperparameters...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

rf_grid_search.fit(X_train, y_train)

print(f"Best parameters: {rf_grid_search.best_params_}")
print(f"Best CV score: {rf_grid_search.best_score_:.3f}")
print()

# Get the best model
best_rf = rf_grid_search.best_estimator_

# 8. Final Model Evaluation
print("8. Evaluating optimized Random Forest...")
y_pred_rf_opt = best_rf.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_opt))
final_mae = mean_absolute_error(y_test, y_pred_rf_opt)
final_r2 = r2_score(y_test, y_pred_rf_opt)

print(f"Optimized Random Forest Results:")
print(f"  RMSE: {final_rmse:.3f}")
print(f"  MAE: {final_mae:.3f}")
print(f"  R²: {final_r2:.3f}")
print()

# 9. Feature Importance Analysis
print("9. Analyzing feature importance...")
feature_importance = best_rf.feature_importances_
feature_names = X.columns

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(importance_df.head(10))
print()

# 10. Final Model Summary
print("10. Final model summary:")
train_r2_rf = r2_score(y_train, best_rf.predict(X_train))
overfitting = abs(train_r2_rf - final_r2)

print(f"Best Model: Optimized Random Forest")
print(f"Test RMSE: {final_rmse:.3f}")
print(f"Test MAE: {final_mae:.3f}")
print(f"Test R²: {final_r2:.3f}")
print(f"Train R²: {train_r2_rf:.3f}")
print(f"Overfitting (R² difference): {overfitting:.3f}")

if overfitting < 0.05:
    print("✅ Overfitting is acceptable (< 5%)")
else:
    print("⚠️ Overfitting is high (> 5%)")

print(f"\nTop 5 Most Important Features:")
for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.3f}")

# 11. Save Model and Results
print("\n11. Saving model and results...")
import joblib
import os
import json

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(best_rf, 'models/best_life_expectancy_model.pkl')
print("Model saved to models/best_life_expectancy_model.pkl")

# Save feature importance
importance_df.to_csv('models/feature_importance.csv', index=False)
print("Feature importance saved to models/feature_importance.csv")

# Save model results summary
results_summary = {
    'model': 'Random Forest (Optimized)',
    'test_rmse': final_rmse,
    'test_mae': final_mae,
    'test_r2': final_r2,
    'train_r2': train_r2_rf,
    'overfitting': overfitting,
    'best_params': rf_grid_search.best_params_
}

with open('models/model_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print("Model results saved to models/model_results.json")

print("\n=== MODELING COMPLETED SUCCESSFULLY ===")
print("Check the 'models' folder for saved files!")
