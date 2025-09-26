import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


plt.style.use('default')
sns.set_palette("husl")


df = pd.read_csv('data/clean_data.csv')
target = 'life_expectancy'


exclude_cols = ['country', 'year', 'status', target]
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


best_rf = joblib.load('models/best_life_expectancy_model.pkl')

os.makedirs('plots', exist_ok=True)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Life Expectancy')
plt.xlabel('Life Expectancy (years)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.boxplot(y)
plt.title('Box Plot of Life Expectancy')
plt.ylabel('Life Expectancy (years)')

plt.tight_layout()
plt.savefig('plots/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()


importance_df = pd.read_csv('models/feature_importance.csv')
top_features = importance_df.head(15)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_features, x='importance', y='feature')
plt.title('Top 15 Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(15, 6))


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_r2 = r2_score(y_test, y_pred_lr)

axes[0].scatter(y_test, y_pred_lr, alpha=0.6, color='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Life Expectancy')
axes[0].set_ylabel('Predicted Life Expectancy')
axes[0].set_title(f'Linear Regression\nR² = {lr_r2:.3f}')


y_pred_rf = best_rf.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)

axes[1].scatter(y_test, y_pred_rf, alpha=0.6, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Life Expectancy')
axes[1].set_ylabel('Predicted Life Expectancy')
axes[1].set_title(f'Random Forest (Optimized)\nR² = {rf_r2:.3f}')

plt.tight_layout()
plt.savefig('plots/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()


residuals = y_test - y_pred_rf

fig, axes = plt.subplots(1, 2, figsize=(15, 6))


axes[0].scatter(y_pred_rf, residuals, alpha=0.6, color='purple')
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Life Expectancy')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Predicted')


axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residuals Distribution')

plt.tight_layout()
plt.savefig('plots/residual_analysis.png', dpi=300, bbox_inches='tight')
plt.close()


correlation_matrix = df[feature_cols + [target]].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()


models_data = {
    'Model': ['Linear Regression', 'Random Forest'],
    'R²': [lr_r2, rf_r2],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_lr)), 
             np.sqrt(mean_squared_error(y_test, y_pred_rf))],
    'MAE': [mean_absolute_error(y_test, y_pred_lr), 
            mean_absolute_error(y_test, y_pred_rf)]
}

comparison_df = pd.DataFrame(models_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))


axes[0].bar(comparison_df['Model'], comparison_df['R²'], color=['blue', 'green'])
axes[0].set_title('R² Comparison')
axes[0].set_ylabel('R² Score')
axes[0].tick_params(axis='x', rotation=45)


axes[1].bar(comparison_df['Model'], comparison_df['RMSE'], color=['red', 'orange'])
axes[1].set_title('RMSE Comparison')
axes[1].set_ylabel('RMSE')
axes[1].tick_params(axis='x', rotation=45)


axes[2].bar(comparison_df['Model'], comparison_df['MAE'], color=['purple', 'brown'])
axes[2].set_title('MAE Comparison')
axes[2].set_ylabel('MAE')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


top_5_features = importance_df.head(5)['feature'].tolist()
top_features_data = df[top_5_features + [target]]

plt.figure(figsize=(12, 8))
correlation_with_target = top_features_data.corr()[target].drop(target).sort_values(ascending=True)
correlation_with_target.plot(kind='barh', color='skyblue')
plt.title('Top 5 Features Correlation with Life Expectancy')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('plots/top_features_correlation.png', dpi=300, bbox_inches='tight')
plt.close()


