from random import shuffle
from tensorflow.keras.datasets import mnist
import tensorflowjs as tfjs
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#print(tf.test.gpu_device_name())

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()

#print(train_features.shape)
#print(test_features.shape)

#plt.imshow(train_features[1,:,:], cmap =plt.cm.binary)
#plt.show()

train_features = train_features.reshape((60000, 28, 28, 1))
test_features = test_features.reshape((10000, 28, 28, 1))

train_features, test_features = train_features / 255, test_features / 255

train_features = train_features.astype("float32")
test_features = test_features.astype("float32")

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


def se(input):
    init = input
    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape((1, 128))(se)
    se = tf.keras.layers.Dense(128//64, activation='relu')(se)
    se = tf.keras.layers.Dense(128, activation='sigmoid')(se)
    x = tf.keras.layers.multiply([init, se])
    return x

s = tf.keras.Input(shape=(28,28,1)) 
model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(s)
model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(model)
model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(model)
model = tf.keras.layers.BatchNormalization()(model)
model = se(model)

model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(model)
model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(model)
model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(model)
model = tf.keras.layers.BatchNormalization()(model)
model = se(model)
model = tf.keras.layers.AveragePooling2D(2)(model)

model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(model)
model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(model)
model = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(model)
model = tf.keras.layers.BatchNormalization()(model)        
model = se(model)
model = tf.keras.layers.AveragePooling2D(2)(model)



model = tf.keras.layers.concatenate([tf.keras.layers.GlobalMaxPooling2D()(model),tf.keras.layers.GlobalAveragePooling2D()(model)])
model = tf.keras.layers.Dense(784, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.00025))(model)
model = tf.keras.layers.Dense(800, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.00025))(model)
model = tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.00025))(model)
model = tf.keras.layers.Dense(10, activation='softmax', use_bias=False, kernel_regularizer=tf.keras.regularizers.l1(0.00025))(model)

with tf.device('/GPU:0'):
    model = tf.keras.Model(inputs=s, outputs=model)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20, width_shift_range=0.1, shear_range=10,
                        height_shift_range=0.1, zoom_range=0.3)
    datagen.fit(train_features)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0005), loss='categorical_crossentropy', metrics = ['accuracy'])
    print(train_features.shape)
    model.fit(datagen.flow(train_features, train_labels, batch_size=64,shuffle=True), epochs=20)
    tfjs.converters.save_keras_model(model,'notSeqv5_20epochs_high_acc_maybe')
    print(model.evaluate(test_features, test_labels)[1])