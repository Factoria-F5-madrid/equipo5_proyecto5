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


plt.style.use('default')
sns.set_palette("husl")


df = pd.read_csv('data/clean_data.csv')


target = 'life_expectancy'



exclude_cols = ['country', 'year', 'status', target]
feature_cols = [col for col in df.columns if col not in exclude_cols]


X = df[feature_cols]
y = df[target]


missing_values = X.isnull().sum()

if missing_values.sum() > 0:
  
    X = X.fillna(X.median())
  
else:
  
print()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
}

initial_results = {}

for name, model in models.items():
   
    model.fit(X_train, y_train)
    
 
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
   
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
  
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

cv_results = {}

for name, model in models.items():
   
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(cv_rmse_scores)
    
    cv_results[name] = {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std()
    }
    
    
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


best_rf = rf_grid_search.best_estimator_


y_pred_rf_opt = best_rf.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_opt))
final_mae = mean_absolute_error(y_test, y_pred_rf_opt)
final_r2 = r2_score(y_test, y_pred_rf_opt)


feature_importance = best_rf.feature_importances_
feature_names = X.columns


importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)


train_r2_rf = r2_score(y_train, best_rf.predict(X_train))
overfitting = abs(train_r2_rf - final_r2)



if overfitting < 0.05:
    print(" Overfitting is acceptable (< 5%)")
else:
    print(" Overfitting is high (> 5%)")


for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
    
import joblib
import os
import json


os.makedirs('models', exist_ok=True)


joblib.dump(best_rf, 'models/best_life_expectancy_model.pkl')

importance_df.to_csv('models/feature_importance.csv', index=False)

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

