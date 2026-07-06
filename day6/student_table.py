import pandas as pd

df = pd.read_csv("D:\\Project\\day6\\students.csv")

print("Shape")
print(df.shape)

print("Columns")
print(df.columns)

print("Data Types")
print(df.dtypes)

print("First 5 Records")
print(df.head())

print("Last 5 Records")
print(df.tail())