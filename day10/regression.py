from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print("Actual")
print(y_test)

print("Predicted")
print(pred)

mse = mean_squared_error(y_test, pred)

print("MSE")
print(mse)

joblib.dump(model, "diabetes_model.pkl")