import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ==========================================
# 1. LOAD DATA & FEATURES
# ==========================================
print("Loading Data...")
try:
    df = pd.read_csv(r'C:\ML Project\Used-Car-Price-Prediction\germany_cars.csv')
except:
    try:
        df = pd.read_csv('germany_cars.csv')
    except FileNotFoundError:
        print("Error: csv file not found.")
        exit()

# Cleaning (same as your other files)
df = df.dropna()
df = df[df['price'] > 500]
df = df[df['price'] < 100000]
df = df[df['year'] > 2000]

# Load features selected by train.py
try:
    selected_features = pickle.load(open('selected_features.pkl', 'rb'))
    print(f"Features loaded: {selected_features}")
except FileNotFoundError:
    print("Error: 'selected_features.pkl' not found. Please run train.py first.")
    exit()

# ==========================================
# 2. PREPARE SHARED DATA SPLIT
# ==========================================
# We use the exact same X_train/X_test for all models to ensure fair comparison
X = df[selected_features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ==========================================
# 3. DEFINE MODELS
# ==========================================
# Setup Preprocessor
categorical_cols = ['make', 'model', 'fuel', 'gear', 'offerType']
chosen_cat_cols = [col for col in selected_features if col in categorical_cols]

ohe = OneHotEncoder(handle_unknown='ignore')
column_trans = make_column_transformer(
    (ohe, chosen_cat_cols),
    remainder='passthrough'
)

# Define the 3 pipelines
models = {
    "Linear Regression": make_pipeline(column_trans, LinearRegression()),
    "Random Forest": make_pipeline(column_trans, RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
    "Gradient Boosting": make_pipeline(column_trans, GradientBoostingRegressor(n_estimators=300, random_state=42))
}

# ==========================================
# 4. TRAIN & EVALUATE LOOP
# ==========================================
results = []

print("\nStarting Comparison...")
for name, pipe in models.items():
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    # Calculate Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({
        "Model": name,
        "R2 Score": r2,
        "MAE (€)": mae
    })

# Create DataFrame for easy plotting
results_df = pd.DataFrame(results)
print("\n--- FINAL RESULTS ---")
print(results_df)

# ==========================================
# 5. PLOTTING COMPARISON
# ==========================================
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: R2 Score (Higher is better)
sns.barplot(data=results_df, x="Model", y="R2 Score", ax=ax1, palette="viridis")
ax1.set_title("Model Accuracy (R2 Score)", fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.0)
for index, row in results_df.iterrows():
    ax1.text(index, row["R2 Score"] + 0.02, f'{row["R2 Score"]:.3f}', color='black', ha="center")

# Plot 2: Mean Absolute Error (Lower is better)
sns.barplot(data=results_df, x="Model", y="MAE (€)", ax=ax2, palette="magma")
ax2.set_title("Mean Absolute Error (Lower is Better)", fontsize=14, fontweight='bold')
for index, row in results_df.iterrows():
    ax2.text(index, row["MAE (€)"] + 50, f'€{row["MAE (€)"]:.0f}', color='black', ha="center")

plt.suptitle("Model Comparison: Linear vs RF vs Gradient Boosting", fontsize=16)
plt.tight_layout()
plt.show()