import pandas as pd

df = pd.read_csv("D:\\Project\\day6\\students.csv")

eligible = df[(df["CGPA"] >= 8.5) &
              (df["Attendance"] >= 90) &
              (df["Backlogs"] == 0)]

print("Eligible Students")
print(len(eligible))

print(eligible[["Name", "Department", "CGPA"]])

print("Department")
print(eligible["Department"].value_counts())

print("Average Marks")
print(eligible["Marks"].mean())

print("Class Average")
print(df["Marks"].mean())

eligible.to_csv("scholarship_shortlist.csv", index=False)
