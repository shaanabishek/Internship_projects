import pandas as pd

students = pd.read_csv("D:\\Project\\day7\\Students.csv")
results = pd.read_csv("D:\\Project\\day7\\Results.csv")
attendance = pd.read_csv("D:\\Project\\day7\\Attendance.csv")

term1 = results[results["Term"] == "Term1"]

marks = term1.groupby("Adm_No")["Marks"].agg(
    Total_Marks="sum",
    Average_Marks="mean"
).reset_index()

attendance["Present_Days"] = attendance["Present_Days"].fillna(0)

attendance["Attendance_Percentage"] = (
    attendance["Present_Days"] / attendance["Working_Days"]
) * 100

att = attendance[["Adm_No", "Attendance_Percentage"]]

final = pd.merge(students, marks, on="Adm_No", how="left")
final = pd.merge(final, att, on="Adm_No", how="left")

print("Report")
print(final)

top = final.sort_values(by="Average_Marks", ascending=False).head(5)

print("Top 5")
print(top)

risk = final[
    (final["Attendance_Percentage"] < 75) &
    (final["Average_Marks"] < 40)
]

print("Risk")
print(risk)

final["Attendance_Percentage"] = final["Attendance_Percentage"].round(2)

final.to_csv("report.csv", index=False)
risk.to_csv("at_risk.csv", index=False)

print("Done")