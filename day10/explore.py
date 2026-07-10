import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame

df["Species"] = df["target"].map(dict(enumerate(iris.target_names)))

print(df.head())

print(df["Species"].value_counts())

print(df.groupby("Species").mean())

avg = df["petal length (cm)"].mean()

filtered = df[
    (df["Species"] == "virginica") &
    (df["petal length (cm)"] > avg)
]

print(filtered)

print(df.groupby("Species")["sepal length (cm)"].mean())

df["Sepal Area"] = df["sepal length (cm)"] * df["sepal width (cm)"]

filtered.to_csv("filtered_iris.csv", index=False)