import numpy as np

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

temp = []

for i in days:
    t = int(input(f"Enter {i} temperature (C): "))
    temp.append(t)

c = np.array(temp)
f = c * 9 / 5 + 32

print("\nTEMPERATURE")

for i in range(len(days)):
    print(days[i], "-", c[i], "C =", round(f[i], 1), "F")