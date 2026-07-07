import pandas as pd

students = pd.read_csv("D:\\Project\\day7\\Students.csv")
results = pd.read_csv("D:\\Project\\day7\\Results.csv")

print("Students")
print(students)

print("Results")
print(results)

print("Duplicates")
print(results["Adm_No"].duplicated().sum())

inner = pd.merge(students, results, on="Adm_No", how="inner")

print("Inner")
print(inner)

inner.to_csv("inner_merge.csv", index=False)

left = pd.merge(students, results, on="Adm_No", how="left")

print("Left")
print(left)

print("No Results")
print(left[left["Marks"].isnull()])

outer = pd.merge(students, results, on="Adm_No", how="outer")

outer["Marks"] = outer["Marks"].fillna(-1)
outer["Subject"] = outer["Subject"].fillna("No Result")

print("Outer")
print(outer)

multi = pd.merge(students, results, on="Adm_No", how="inner")

print("Merge")
print(multi)