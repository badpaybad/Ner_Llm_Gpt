import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

import keras
from keras import layers

_arr28_28 = tf.constant([[0, 1, 2, 3, 4, 5, 5, 34, 13, 32]])

arr1 = tf.constant([[0, 1, 12, 13, 4, 5, 45, 3, 43, 32]])

print(tf.reshape(arr1,(-1,1,10)))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(1, 10)))
#model.add(tf.keras.layers.Flatten())
model.add(layers.Conv1D(3, 1, activation="relu"))

print("--->>> model(tf.reshape(arr1,(-1,1,10)))")
print(model(tf.reshape(arr1,(-1,1,10))))