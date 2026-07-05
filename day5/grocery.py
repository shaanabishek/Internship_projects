import numpy as np

items = ["Rice", "Milk", "Bread", "Eggs", "Juice"]

price = np.array([60, 45, 35, 7, 120])
qty = np.array([2, 3, 2, 12, 1])

total = price * qty

print("GROCERY BILL")

for i in range(len(items)):
    print(items[i], "-", qty[i], "x", price[i], "=", total[i])

print("\nGrand Total:", total.sum())