import os
import joblib
import pickle
import glob

# 1. Find the large file automatically
print("Searching for .pkl files over 100MB...")
pkl_files = glob.glob("*.pkl")
large_file = None

for f in pkl_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"Found: {f} ({size_mb:.2f} MB)")
    if size_mb > 100:
        large_file = f
        break

if not large_file:
    print("\n❌ ERROR: No file over 100MB found.")
    print("Please check if your model file is actually in this folder.")
    input("Press Enter to exit...")
    exit()

print(f"\n✅ Large file found: {large_file}")
print("Attempting compression...")

# 2. Compress the file
try:
    with open(large_file, "rb") as f:
        model = pickle.load(f)
    
    output_filename = "model_final.pkl"
    joblib.dump(model, output_filename, compress=9)
    
    new_size = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"\nSUCCESS! Compressed to: {new_size:.2f} MB")
    print(f"Created new file: {output_filename}")
    print("-" * 30)
    print("NEXT STEPS (Run these in terminal):")
    print(f"1. del {large_file}")
    print(f"2. rename {output_filename} {large_file}")
    print("-" * 30)

except Exception as e:
    print(f"\n❌ Error during compression: {e}")
    # If pickle fails, it might be because joblib is needed to LOAD it too
    try:
        print("Trying to load with joblib instead...")
        model = joblib.load(large_file)
        joblib.dump(model, "model_final.pkl", compress=9)
        print("Success with joblib!")
    except:
        print("Failed. The file might be corrupted.")

input("Press Enter to exit...")