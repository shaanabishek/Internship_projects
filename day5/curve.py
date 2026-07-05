import numpy as np

names = ["Arun", "Nisha", "Rahul", "Priya", "Karthik", "Sneha"]

scores = np.array([68, 74, 59, 87, 79, 65])

new = scores + 5

print("EXAM CURVE")

for n, s, i in zip(names, scores, new):
    print(n, ":", s, "->", i)

print()
print("Original Mean:", round(scores.mean(), 2))
print("New Mean:", round(new.mean(), 2))