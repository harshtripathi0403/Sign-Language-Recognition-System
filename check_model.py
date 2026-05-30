import joblib

model = joblib.load("02_repo/isl_model.pkl")

print(type(model))
print("Has attribute n_estimators:", hasattr(model, "n_estimators"))

try:
    print("Feature importances:", model.feature_importances_[:5])
except Exception as e:
    print("ERROR:", e)