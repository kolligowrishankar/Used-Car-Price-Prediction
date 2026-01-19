import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
        print("Error: csv file not found. Please check the path.")
        exit()

# ==========================================
# 2. DATA CLEANING
# ==========================================
df = df.dropna()
df = df[df['price'] > 500]
df = df[df['price'] < 100000]
df = df[df['year'] > 2000]

# ==========================================
# 3. LOAD SELECTED FEATURES FROM TRAIN.PY
# ==========================================
# This is the critical step that ensures we use the EXACT same features
# that Gradient Boosting found to be important.
try:
    selected_features = pickle.load(open('selected_features.pkl', 'rb'))
    print(f"\nSUCCESS: Loaded {len(selected_features)} selected features from train.py")
    print(f"Features: {selected_features}")
except FileNotFoundError:
    print("\nERROR: 'selected_features.pkl' not found!")
    print("STEP 1: Run 'train.py' first to select the best features.")
    print("STEP 2: Then run this script again.")
    exit()

# ==========================================
# 4. PREPARE DATA
# ==========================================
X = df[selected_features]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ==========================================
# 5. SETUP PIPELINE (Random Forest)
# ==========================================
# We only encode columns that are actually in our selected list
categorical_cols = ['make', 'model', 'fuel', 'gear', 'offerType']
chosen_cat_cols = [col for col in selected_features if col in categorical_cols]

ohe = OneHotEncoder(handle_unknown='ignore')
column_trans = make_column_transformer(
    (ohe, chosen_cat_cols),
    remainder='passthrough'
)

# Random Forest with optimized settings
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
pipe = make_pipeline(column_trans, rf)

# ==========================================
# 6. TRAIN & EVALUATE
# ==========================================
print("\nTraining Random Forest Model...")
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Calculate Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"------------------------------------------------")
print(f"Random Forest Results:")
print(f"R2 Score (Accuracy): {r2:.4f}")
print(f"Mean Absolute Error: €{mae:.2f}")
print(f"------------------------------------------------")

# Save the model
pickle.dump(pipe, open('model_rf.pkl', 'wb'))
print("Model saved as 'model_rf.pkl'")

# ==========================================
# 7. PLOTTING (3 Graphs on One Page)
# ==========================================
print("Generating Analysis Plots...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# --- Plot 1: Accuracy (Actual vs Predicted) ---
ax1.scatter(y_test, y_pred, alpha=0.3, color='green')
# Draw the "perfect prediction" red dashed line
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_title(f"RF Accuracy (R2: {r2:.2f})")
ax1.set_xlabel("Actual Price (€)")
ax1.set_ylabel("Predicted Price (€)")
ax1.grid(True, alpha=0.3)

# --- Plot 2: Residuals (Error Distribution) ---
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=ax2, color='green', bins=30)
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_title("Residual Distribution (Errors)")
ax2.set_xlabel("Error (Actual - Predicted)")
ax2.set_ylabel("Frequency")
ax2.grid(True, alpha=0.3)

# --- Plot 3: Feature Contribution ---
# We calculate importance specifically for this Random Forest model
# to see how it used the features provided by train.py
print("Calculating feature contribution (this may take a moment)...")
perm_result = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=42)

imp_df = pd.DataFrame({
    'Feature': selected_features,
    'Contribution': perm_result.importances_mean
}).sort_values(by='Contribution', ascending=True)

ax3.barh(imp_df['Feature'], imp_df['Contribution'], color='darkgreen')
ax3.set_title("Feature Contribution (Random Forest)")
ax3.set_xlabel("Impact on Model Accuracy")
ax3.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
print("Done! Plots displayed.")