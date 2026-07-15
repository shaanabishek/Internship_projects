from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Default Model

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

pred = knn.predict(x_test)

acc1 = accuracy_score(y_test, pred)

print("Default Accuracy")
print(acc1)

# GridSearchCV

param = {
    "n_neighbors": [3, 5, 7, 9, 11]
}

grid = GridSearchCV(KNeighborsClassifier(), param, cv=5)

grid.fit(x_train, y_train)

print("Grid Best Parameter")
print(grid.best_params_)

print("Grid Best Score")
print(grid.best_score_)

best_knn = grid.best_estimator_

pred2 = best_knn.predict(x_test)

acc2 = accuracy_score(y_test, pred2)

print("Grid Accuracy")
print(acc2)

# RandomizedSearchCV

random = RandomizedSearchCV(
    KNeighborsClassifier(),
    param,
    cv=5,
    n_iter=5,
    random_state=42
)

random.fit(x_train, y_train)

print("Random Best Parameter")
print(random.best_params_)

print("Random Best Score")
print(random.best_score_)

best_random = random.best_estimator_

pred3 = best_random.predict(x_test)

acc3 = accuracy_score(y_test, pred3)

print("Random Accuracy")
print(acc3)

# Comparison

print("Comparison")
print("Default :", acc1)
print("Grid :", acc2)
print("Random :", acc3)

# Conclusion

if acc2 >= acc3 and acc2 >= acc1:
    print("GridSearchCV performed better because it checked all values.")
elif acc3 >= acc2 and acc3 >= acc1:
    print("RandomizedSearchCV performed better because it found a good value quickly.")
else:
    print("The default model performed best.")