import pandas as pd

df = pd.read_csv("D:\\Project\\day7\\Student_results.csv")

print("Shape")
print(df.shape)

print("First 5")
print(df.head())

print("Percentage")
per = df["Percentage"]
print(per)

high = df[df["Percentage"] >= 75]

print("75+")
print(high)

high.to_csv("D:\\Project\\day7\\highest.csv", index=False)

low = df[(df["Subject"] == "Maths") & (df["Marks"] < 40)]

print("Maths < 40")
print(low)

print("Count")
print(len(low))

c2 = df.query("Class == 'II' and Percentage > 60")

print("Class II")
print(c2)

print("Average")
print(high["Percentage"].mean())

print("Names")
print(low[["Name", "Marks"]])