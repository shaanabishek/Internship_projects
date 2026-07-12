import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ---------------- Logistic Regression ----------------

lr = LogisticRegression(max_iter=5000)

lr.fit(x_train, y_train)

pred1 = lr.predict(x_test)

acc1 = accuracy_score(y_test, pred1)

print("Logistic Regression")
print("Accuracy")
print(acc1)

print("Confusion Matrix")
print(confusion_matrix(y_test, pred1))

print("Classification Report")
print(classification_report(y_test, pred1))

# ---------------- KNN ----------------

score = []

for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    acc = accuracy_score(y_test, pred)
    score.append(acc)

    print("K =", k)
    print(acc)

best = [3, 5, 7][score.index(max(score))]
print("Best K")
print(best)

knn = KNeighborsClassifier(n_neighbors=best)
knn.fit(x_train, y_train)

pred2 = knn.predict(x_test)

acc2 = accuracy_score(y_test, pred2)

print("Confusion Matrix")
print(confusion_matrix(y_test, pred2))

print("Classification Report")
print(classification_report(y_test, pred2))

# ---------------- Decision Tree ----------------

dt = DecisionTreeClassifier(random_state=42)

dt.fit(x_train, y_train)

pred3 = dt.predict(x_test)

acc3 = accuracy_score(y_test, pred3)

print("Decision Tree")
print("Accuracy")
print(acc3)

print("Confusion Matrix")
print(confusion_matrix(y_test, pred3))

print("Classification Report")
print(classification_report(y_test, pred3))

# ---------------- Comparison ----------------

table = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Decision Tree"],
    "Accuracy": [
        acc1,
        acc2,
        acc3
    ],
    "Precision": [
        precision_score(y_test, pred1),
        precision_score(y_test, pred2),
        precision_score(y_test, pred3)
    ],
    "Recall": [
        recall_score(y_test, pred1),
        recall_score(y_test, pred2),
        recall_score(y_test, pred3)
    ],
    "F1-Score": [
        f1_score(y_test, pred1),
        f1_score(y_test, pred2),
        f1_score(y_test, pred3)
    ]
})

print("Comparison")
print(table)

# ---------------- New Data ----------------

new = x_test[:5]

print("Logistic Regression")
print(lr.predict(new))

print("KNN")
print(knn.predict(new))

print("Decision Tree")
print(dt.predict(new))

# ---------------- Best Model ----------------

best_acc = max(acc1, acc2, acc3)

if best_acc == acc1:
    model = "Logistic Regression"
elif best_acc == acc2:
    model = "KNN"
else:
    model = "Decision Tree"

print("Best Model")
print(model)

print("Best Accuracy")
print(best_acc)

print("Reason")
print("This model has the highest accuracy among all models.")
print("It gives better predictions on the test dataset.")