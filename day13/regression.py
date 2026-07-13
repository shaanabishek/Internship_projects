import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = load_diabetes()

x = data.data[:, [2]]
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print("Actual")
print(y_test[:10])

print("Predicted")
print(pred[:10])

plt.scatter(x_test, y_test)
plt.plot(x_test, pred)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

m = model.coef_[0]
c = model.intercept_

print("Slope")
print(m)

print("Intercept")
print(c)

print("Equation")
print("y =", m, "x +", c)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MAE")
print(mae)

print("MSE")
print(mse)

print("R2 Score")
print(r2)

new = [[0.05], [0.08], [0.10]]

print("New Prediction")
print(model.predict(new))

print("Performance")

if r2 > 0.8:
    print("Best")
    print("High R2 score and low error.")
elif r2 > 0.5:
    print("Good")
    print("Model predicts reasonably well.")
else:
    print("Needs Improvement")
    print("Low R2 score and higher prediction error.")