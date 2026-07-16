import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# Load Dataset
data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = data.flow_from_directory(
    "D:/Project/day16/archive/Car-Bike-Dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

test = data.flow_from_directory(
    "D:/Project/day16/archive/Car-Bike-Dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# CNN Model
model = Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(1,activation="sigmoid"))

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train,
    validation_data=test,
    epochs=10
)

# Save Model
model.save("vehicle_model.h5")

# Evaluate
loss, acc = model.evaluate(test)

print("Test Accuracy")
print(acc)

# Check Overfitting
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print("Training Accuracy")
print(train_acc)

print("Validation Accuracy")
print(val_acc)

if train_acc - val_acc > 0.1:
    print("Model is Overfitting")
else:
    print("Model is Good")

# Plot Accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["Train","Validation"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Test Custom Images
images = [
    "D:/Project/day16/test/car1.jpg",
    "D:/Project/day16/test/car2.jpg",
    "D:/Project/day16/test/bike1.jpg",
    "D:/Project/day16/test/bike2.jpg",
    "D:/Project/day16/test/car3.jpg"
]

print("Predictions")

for file in images:

    img = load_img(file, target_size=(150,150))
    img = img_to_array(img)
    img = np.expand_dims(img,0)
    img = img/255

    pred = model.predict(img, verbose=0)

    if pred[0][0] > 0.5:
        print(file, "Car")
    else:
        print(file, "Bike")

# Layer Explanation
print("\nConv2D")
print("Extracts important features from images.")

print("\nMaxPooling2D")
print("Reduces image size while keeping useful features.")

print("\nFlatten")
print("Converts feature maps into a single vector.")

print("\nDense")
print("Classifies the image based on extracted features.")

print("\nDropout")
print("Reduces overfitting by randomly dropping neurons.")

# Summary
print("\nSummary")
print("Dataset : Car vs Bike")
print("Epochs : 10")
print("Model : CNN")
print("Model Saved : vehicle_model.h5")