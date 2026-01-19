import pickle
import joblib
import os

# 1. Load the large model
print("Loading large model...")
with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Save it with maximum compression (Level 9)
print("Compressing and saving...")
joblib.dump(model, "model_rf_compressed.pkl", compress=9)

# 3. Check sizes
old_size = os.path.getsize("model_rf.pkl") / (1024 * 1024)
new_size = os.path.getsize("model_rf_compressed.pkl") / (1024 * 1024)

print(f"Original Size: {old_size:.2f} MB")
print(f"Compressed Size: {new_size:.2f} MB")
print("Done! You can now use the compressed file.")