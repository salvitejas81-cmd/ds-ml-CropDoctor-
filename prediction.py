import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.pkl")


def patch_keras_loading():
    original_glorot_from_config = tf.keras.initializers.GlorotUniform.from_config

    @classmethod
    def patched_glorot_from_config(cls, config):
        config = dict(config)
        config.pop("input_axes", None)
        config.pop("output_axes", None)
        return original_glorot_from_config(config)

    tf.keras.initializers.GlorotUniform.from_config = patched_glorot_from_config

    original_dense_from_config = tf.keras.layers.Dense.from_config

    @classmethod
    def patched_dense_from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return original_dense_from_config(config)

    tf.keras.layers.Dense.from_config = patched_dense_from_config


patch_keras_loading()

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

with open(CLASS_NAMES_PATH, "rb") as file:
    class_indices = pickle.load(file)

class_names = [
    name
    for name, index in sorted(class_indices.items(), key=lambda item: item[1])
]


def predict_disease(img_path):
    img = image.load_img(
        img_path,
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction) * 100)

    return predicted_class, round(confidence, 2)