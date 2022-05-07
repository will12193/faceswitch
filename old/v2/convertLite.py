import tensorflow as tf
import sys

path = sys.argv[1]

model = tf.keras.models.load_model(path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

