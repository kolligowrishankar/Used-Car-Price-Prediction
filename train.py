import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")
try:
    df = pd.read_csv('C:\ML Project\German cars\germany_cars.csv')
except:
    try:
        df = pd.read_csv('germany_cars.csv') 
    except FileNotFoundError:
        print("Error: csv file not found.")
        exit()

# ==========================================
# 2. DATA CLEANING
# ==========================================
df = df.dropna()
df = df[df['price'] > 500]
df = df[df['price'] < 100000]
df = df[df['year'] > 2000]

# ==========================================
# 3. SETUP MODEL FOR SELECTION
# ==========================================
all_features = ['make', 'model', 'year', 'mileage', 'hp', 'fuel', 'gear', 'offerType']
X = df[all_features]
y = df['price']

# Basic Pipeline for Selection
ohe = OneHotEncoder(handle_unknown='ignore')
categorical_cols = ['make', 'model', 'fuel', 'gear', 'offerType']

column_trans = make_column_transformer(
    (ohe, categorical_cols),
    remainder='passthrough'
)

gb = GradientBoostingRegressor(n_estimators=300, random_state=42)
pipe = make_pipeline(column_trans, gb)

# ==========================================
# 4. TRAIN AND CALCULATE IMPORTANCE
# ==========================================
print("Training Gradient Boosting to find important features...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
pipe.fit(X_train, y_train)

print("Calculating Feature Importance...")
result = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=42)

# ==========================================
# 5. SELECT FEATURES (> 1% importance)
# ==========================================
importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Ranking ---")
print(importance_df)

selected_features = importance_df[importance_df['Importance'] > 0.01]['Feature'].tolist()
print(f"\nSELECTED FEATURES: {selected_features}")

# ==========================================
# 6. SAVE SELECTED FEATURES
# ==========================================
pickle.dump(selected_features, open('selected_features.pkl', 'wb'))
print("Success: Feature list saved to 'selected_features.pkl'")

# ==========================================
# 7. RETRAIN FINAL MODEL WITH SELECTED FEATURES
# ==========================================
print("\nRetraining Gradient Boosting with ONLY selected features...")
X_selected = df[selected_features]
X_train_sel, X_test_sel, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)

# New Pipeline for selected features
chosen_cat_cols = [col for col in selected_features if col in categorical_cols]
final_trans = make_column_transformer((ohe, chosen_cat_cols), remainder='passthrough')
final_pipe = make_pipeline(final_trans, gb)

final_pipe.fit(X_train_sel, y_train)
y_pred = final_pipe.predict(X_test_sel)

# ==========================================
# 8. METRICS (MAE & R2)
# ==========================================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  # <--- ADDED MAE HERE

print(f"------------------------------------------------")
print(f"Gradient Boosting Results:")
print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error: €{mae:.2f}")  # <--- PRINTING MAE HERE
print(f"------------------------------------------------")

pickle.dump(final_pipe, open('model_gb.pkl', 'wb'))
print("Model saved as 'model_gb.pkl'")

# ==========================================
# 9. PLOTTING (3 Graphs on One Page)
# ==========================================
print("Generating Plots...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Accuracy (Actual vs Predicted)
ax1.scatter(y_test, y_pred, alpha=0.3, color='blue')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_title(f"GB Accuracy (R2: {r2:.2f})")
ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")
ax1.grid(True)

# Plot 2: Residuals (Histogram)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=ax2, color='blue', bins=30)
ax2.axvline(0, color='red', linestyle='--')
ax2.set_title("Residual Distribution (Errors)")
ax2.set_xlabel("Error (Actual - Predicted)")
ax2.grid(True)

# Plot 3: Feature Contribution
# Recalculate importance specifically for the FINAL model
final_perm_result = permutation_importance(final_pipe, X_test_sel, y_test, n_repeats=10, random_state=42)
final_imp_df = pd.DataFrame({
    'Feature': selected_features,
    'Contribution': final_perm_result.importances_mean
}).sort_values(by='Contribution', ascending=True)

ax3.barh(final_imp_df['Feature'], final_imp_df['Contribution'], color='teal')
ax3.set_title("Feature Contribution (Gradient Boosting)")
ax3.set_xlabel("Impact on Model Accuracy")
ax3.grid(True, axis='x')

plt.tight_layout()
plt.show()
print("Done! Plots displayed.")