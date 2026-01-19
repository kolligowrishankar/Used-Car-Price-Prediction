import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# 1. LOAD DATA
try:
    df = pd.read_csv('C:\ML Project\German cars\germany_cars.csv')
except:
    df = pd.read_csv('germany_cars.csv')

df = df.dropna()
df = df[df['price'] > 500]
df = df[df['price'] < 100000]
df = df[df['year'] > 2000]

# 2. LOAD SELECTED FEATURES
try:
    selected_features = pickle.load(open('selected_features.pkl', 'rb'))
    print(f"Using features: {selected_features}")
except FileNotFoundError:
    print("Error: Run train.py first!")
    exit()

# 3. PREPARE DATA
X = df[selected_features]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SETUP PIPELINE
categorical_cols = ['make', 'model', 'fuel', 'gear', 'offerType']
chosen_cat_cols = [col for col in selected_features if col in categorical_cols]

ohe = OneHotEncoder(handle_unknown='ignore')
column_trans = make_column_transformer(
    (ohe, chosen_cat_cols),
    remainder='passthrough'
)

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)

# 5. TRAIN
print("Training Linear Regression...")
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"Linear Regression R2 Score: {r2:.4f}")

pickle.dump(pipe, open('model_linear.pkl', 'wb'))

# ==========================================
# 6. PLOTTING (3 Graphs on One Page)
# ==========================================
print("Generating Plots...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Accuracy
ax1.scatter(y_test, y_pred, alpha=0.3, color='orange')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_title(f"LR Accuracy (R2: {r2:.2f})")
ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")
ax1.grid(True)

# Plot 2: Residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=ax2, color='orange')
ax2.axvline(0, color='red', linestyle='--')
ax2.set_title("Residual Distribution (Errors)")
ax2.set_xlabel("Error (Actual - Predicted)")
ax2.grid(True)

# Plot 3: Feature Contribution
perm_result = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=42)
imp_df = pd.DataFrame({
    'Feature': selected_features,
    'Contribution': perm_result.importances_mean
}).sort_values(by='Contribution', ascending=True)

ax3.barh(imp_df['Feature'], imp_df['Contribution'], color='darkorange')
ax3.set_title("Feature Contribution (Linear Regression)")
ax3.set_xlabel("Impact on Model Accuracy")
ax3.grid(True, axis='x')

plt.tight_layout()
plt.show()