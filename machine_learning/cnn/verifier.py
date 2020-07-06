import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input
IMG_SIZE = 160
BATCH_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_learning_rate = 0.005

def create_model():
    # Create the base model from the pre-trained model MobileNet V2
    base_model1 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # keep layer names unique
    for l in base_model1.layers:
        l._name = l._name+'_1'

    base_model1.trainable = True
    base_model2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model2.trainable = True

    concatenate = Concatenate()([base_model1.output, base_model2.output])
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(concatenate)
    #dense_layer = tf.keras.layers.Dense(50, activation='tanh')(global_average_layer)
    dense_layer = tf.keras.layers.Dense(2, activation='tanh')(global_average_layer)
    prediction_layer = tf.keras.layers.Softmax()(dense_layer)
    model = Model(inputs=[base_model1.inputs, base_model2.inputs], outputs=prediction_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

# model = create_model()
# x = np.zeros([1, IMG_SIZE, IMG_SIZE, 3])
# out = model.predict([x, x])
# print(out)