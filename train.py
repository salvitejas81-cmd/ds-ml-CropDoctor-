import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "Dataset", "PlantVillage", "train")
val_path = os.path.join(BASE_DIR, "Dataset", "PlantVillage", "val")

train_data = ImageDataGenerator(rescale=1 / 255)
val_data = ImageDataGenerator(rescale=1 / 255)

train_dataset = train_data.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

val_dataset = val_data.flow_from_directory(
    val_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

print("Training Images:", train_dataset.samples)
print("Validation Images:", val_dataset.samples)
print("Classes:")
print(train_dataset.class_indices)

num_classes = len(train_dataset.class_indices)

model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    verbose=1
)

model.save(os.path.join(BASE_DIR, "plant_disease_model.keras"))

with open(os.path.join(BASE_DIR, "class_names.pkl"), "wb") as file:
    pickle.dump(train_dataset.class_indices, file)

print("Model Saved Successfully!")
print("Class Names Saved Successfully!")
print("Training Accuracy:", history.history["accuracy"][-1])
print("Validation Accuracy:", history.history["val_accuracy"][-1])