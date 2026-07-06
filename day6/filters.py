import pandas as pd

df = pd.read_csv("D:\\Project\\day6\\students.csv")

print("1")
s1 = df[(df["Marks"] > 80) & (df["Backlogs"] == 0)]
print(s1)

print("2")
s2 = df[(df["Department"] == "CSE") & (df["Year"] == "3rd Year")]
print(s2)

print("3")
avg = df["CGPA"].mean()
print(df[df["CGPA"] > avg])