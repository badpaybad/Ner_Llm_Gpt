import numpy as np
import os

#os.environ["KERAS_BACKEND"] = "jax"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# # Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255

numtoTrain=5000

x_train=x_train[:numtoTrain]
y_train=y_train[:numtoTrain]
x_test=x_test[:numtoTrain]
y_test=y_test[:numtoTrain]
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print("x_train[0]")
print(x_train[0].shape)

def resize_fn (x):
    #temp =keras.layers.Resizing(150, 150)(tf.constant( x))
    temp =keras.layers.Resizing(150, 150)(x)
    temp=tf.tile(temp, [1, 1, 3]) 
    return temp
    
if os.path.isfile("finetune.resnet50.keras")==False and os.path.isdir("finetune.resnet50.keras")==False:

    # Model parameters
    num_classes = len(set(y_train))
    input_shape = (28, 28, 1)
    print(y_train)
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    # print(y_train)
    
    base_model = keras.applications.ResNet50(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3), 
        #classes=num_classes,
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = True

    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))

    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    # x = scale_layer(inputs)
    x=inputs
    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=True)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(num_classes)(x)
    model = keras.Model(inputs, outputs)

    model.summary(show_trainable=True)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    epochs = 2
    from tensorflow import data as tf_data

    # train_ds, validation_ds, test_ds = tfds.load(
    #     "cats_vs_dogs",
    #     # Reserve 10% for validation and 10% for test
    #     split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    #     as_supervised=True,  # Include labels
    # )

    # train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
    # validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
    # test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))

    # batch_size = 64

    # train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
    # validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
    # test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()

#tensorboard --logdir=logsfinetune
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath="finetune_model_at_epoch_{epoch}.keras"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        TensorBoard(log_dir='logsfinetune', histogram_freq=1)
    ]
    totrain=np.array([resize_fn(x) for x in x_train])
    print("Fitting the top layer of the model")
    model.fit(totrain  , y_train, epochs=epochs,               
        batch_size=100,        
        validation_split=0.15,
                          #validation_data=validation_ds,                          
        callbacks=callbacks,
                          )
    score = model.evaluate(np.array([resize_fn(x) for x in x_test]) , y_test, verbose=0)
    model.save("finetune.resnet50.keras")

model = keras.saving.load_model("finetune.resnet50.keras")
# print("score = model.evaluate(np.array([resize_fn(x) for x in x_test]) , y_test, verbose=0)")

# print( model.evaluate(np.array([resize_fn(x) for x in x_test]) , y_test, verbose=0))
print("-----")
for i in range(10):
    testPredict1 =x_train[i]
    testPredict1 =keras.layers.Resizing(150, 150)(tf.constant(testPredict1))
    #
    #testPredict1 =keras.layers.Resizing(150, 150)(x_train[i])
    testPredict1=tf.tile(testPredict1, [1, 1, 3]) 
    
    testPredict1=tf.expand_dims(testPredict1, axis=0)
    
    predictions = model.predict(testPredict1)

    print(f"Img: {i} similar to idx: {np.argmax(predictions[0])} score: {np.amax(predictions[0])}")
    # import math
    # for idx, p in enumerate(predictions[0]):
    #     print(f"{idx} -> {round(p,3)}")
