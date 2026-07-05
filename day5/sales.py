import numpy as np

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

sales = np.array([5200, 6100, 4700, 8000, 7200, 6900, 5400])

avg = sales.mean()

print("WEEKLY SALES")

for i in range(len(days)):
    print(days[i], "-", sales[i])

print("\nAverage:", round(avg, 2))
print("Highest:", sales.max())
print("Lowest:", sales.min())
print("Total:", sales.sum())

count = np.sum(sales > avg)

print("Above Average Days:", count)