import joblib; import pickle; print('Loading...'); model = pickle.load(open('model_rf.pkl', 'rb')); print('Compressing...'); joblib.dump(model, 'model_compressed.pkl', compress=9); print('Done!') 
