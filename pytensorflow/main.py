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

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(1, 10)))
# model.add (tf.keras.layers.Flatten(input_shape=(, )))
# model.add(  tf.keras.layers.Softmax())
# model.add(  tf.keras.layers.Dense(10, activation=lambda x: tf.keras.activations.relu(x,max_value=None)))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

_arr28_28 = [[0, 1, 2, 3, 4, 5, -5, -4, -3, -2]]

arr1 = [[0, 1, -2, -3, 4, 5, -5, 3, -3, -2]]

input_tf=tf.constant([_arr28_28,arr1])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(input_tf, tf.constant([0,1]), epochs=5)

# print("model(input_tf)")
# print(model(input_tf))
# # the same to call predict
result = model.predict(tf.constant([_arr28_28]))
print("result = model.predict(tf.constant([_arr28_28]))")
print(result)
import numpy
print(numpy.argmax(result[0]))
print(numpy.amax(result[0]))

exit(0)

if __name__ == "__main__":

    print(f"tf.__version__: {tf.__version__}")

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    predictions = model(x_train[:1]).numpy()
    tf.nn.softmax(predictions).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    probability_model(x_test[:5])
