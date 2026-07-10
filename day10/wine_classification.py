from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()

x = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

acc = accuracy_score(y_test, pred)

print("Accuracy")
print(acc)