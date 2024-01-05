import numpy
from keras.callbacks import TensorBoard
import keras
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
# cpu only
# python3 -m pip install tensorflow-cpu
# python3 -m pip install --upgrade pip
# pip3 install -U tensorflow-cpu
# https://www.tensorflow.org/install/pip#package-location\

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

_arrDataTrain=[]
_lbl=[]

for i in range(0,1000):
    _arrDataTrain.append([[1, i, 2, 3, 4, 5, int(i*2), 4, 3, 2]])    
    _lbl.append(0)


for i in range(0,1000):
    _arrDataTrain.append([[0, -1, int(i*-1), -3, -4, -5, int(i*-2), -3, -3, -2]])    
    _lbl.append(1)

print("_lbl.__len__()")
print(_lbl.__len__())
print(set(_lbl).__len__())

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(1, 10)))
model.add(tf.keras.layers.Softmax())
model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
# # model.add(  tf.keras.layers.Dense(10, activation=lambda x: tf.keras.activations.relu(x,max_value=None)))
#model.add(tf.keras.layers.GlobalAveragePooling1D())
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(set(_lbl).__len__(), activation="softmax"))


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file="test_model.png")
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


history = model.fit(tf.constant(_arrDataTrain), tf.constant(_lbl), epochs=100, callbacks=[
    TensorBoard(log_dir='logs', histogram_freq=1)
]
)
# to show log tensorboard --logdir=logs

_arrTest = [[-1, -1, -2, -3, -4, -5, -5, -4, -3, -2]]
_arrTest = [[1, 1, 2, 3, 4, 5, 5, 4, 3, 2]]
# print("model(input_tf)")
# print(model(input_tf))
# # the same to call predict
result = model.predict(tf.constant([_arrTest]))
print("result = model.predict(tf.constant([_arr28_28]))")
print(result)
print(numpy.argmax(result[0]))
print(numpy.amax(result[0]))

for idx,v in enumerate(result[0]):
    print(f"{idx} -> {round(v,3)}")
