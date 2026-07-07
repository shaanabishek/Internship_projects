import pandas as pd

df = pd.read_csv("D:\\Project\\day7\\Exam_scores.csv")

print("Info")
print(df.info())

print("Missing")
print(df.isnull().sum())

df["Marks"] = df.groupby("Subject")["Marks"].transform(
    lambda x: x.fillna(x.mean())
)

print("After Fill")
print(df.isnull().sum())

sub = df.groupby("Subject")["Marks"].agg(["count", "mean", "min", "max"])

print("Subject")
print(sub)

stu = df.groupby("Name")["Marks"].mean()

print("Student")
print(stu)

top = df.groupby("Name")["Marks"].sum().sort_values(ascending=False).head(3)

print("Top 3")
print(top)

print("Reset")
print(sub.reset_index())