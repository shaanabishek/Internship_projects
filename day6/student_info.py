import pandas as pd

df = pd.read_csv("D:\\Project\\day6\\students.csv")

print("Highest Marks")
print(df[df["Marks"] == df["Marks"].max()])

print("Average CGPA")
print(df["CGPA"].mean())

print("Average Attendance")
print(df["Attendance"].mean())