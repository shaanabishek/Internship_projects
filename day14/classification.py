from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

data = load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, pred))

acc = accuracy_score(y_test, pred)
pre = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("Accuracy")
print(acc)

print("Precision")
print(pre)

print("Recall")
print(rec)

print("F1 Score")
print(f1)

print("Classification Report")
print(classification_report(y_test, pred))

score = cross_val_score(model, x, y, cv=5)

print("Cross Validation Scores")
print(score)

print("Mean Accuracy")
print(score.mean())

print("Performance")

if score.mean() > 0.9:
    print("Best")
    print("The model has high accuracy and performs consistently.")
elif score.mean() > 0.8:
    print("Good")
    print("The model performs well but can be improved.")
else:
    print("Needs Improvement")
    print("The model accuracy is low.")