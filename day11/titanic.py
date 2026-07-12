import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("D:\\Project\\day11\\Titanic.csv")

print("First 5")
print(df.head())

print("Last 5")
print(df.tail())

print("Shape")
print(df.shape)

print("Info")
print(df.info())

df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

x = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

print("X")
print(x)

print("Y")
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("Train Shape")
print(x_train.shape)

print("Test Shape")
print(x_test.shape)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print("Predicted")
print(pred[:10])

print("Actual")
print(y_test.values[:10])

new = [
    [3, 0, 22, 7.25],
    [1, 1, 38, 71.28],
    [2, 0, 30, 15.50]
]

print("New Prediction")
print(model.predict(new))

print("Summary")
print("Dataset : Titanic")
print("Records :", len(df))
print("Features :", x.shape[1])
print("Target : Survived")
print("Model : Decision Tree")