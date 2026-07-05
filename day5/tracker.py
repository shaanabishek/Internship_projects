import numpy as np

data = {}

try:
    f = open("steps_log.txt", "r")

    for line in f:
        d, s = line.strip().split(":")
        data[d] = int(s)

    f.close()

except:
    print("File not found")
    exit()

days = list(data.keys())
steps = np.array(list(data.values()))

print("FITTRACK")

for i in range(len(days)):
    print(days[i], "-", steps[i], "steps")

print("\nAverage:", round(np.mean(steps), 2))
print("Highest:", np.max(steps))
print("Lowest:", np.min(steps))

goal = np.sum(steps >= 8000)
print("Goal %:", round((goal / len(steps)) * 100, 2))

print("Most Active:", days[np.argmax(steps)])
print("Least Active:", days[np.argmin(steps)])

print("\nRanking")

order = np.argsort(steps)[::-1]

for i in range(len(order)):
    print(i + 1, ".", days[order[i]], "-", steps[order[i]])