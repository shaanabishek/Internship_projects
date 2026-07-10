import joblib

model = joblib.load("diabetes_model.pkl")

new = [[
    0.05, -0.04, 0.06, 0.02, -0.03,
    -0.04, 0.01, -0.02, 0.04, 0.03
]]

pred = model.predict(new)

print("Prediction")
print(pred)

print("Model loaded successfully")