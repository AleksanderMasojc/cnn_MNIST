from keras.datasets import mnist
import tensorflowjs as tfjs
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()

#print(train_features.shape)
#print(test_features.shape)

#plt.imshow(train_features[1,:,:], cmap =plt.cm.binary)
#plt.show()

train_features = train_features.reshape((60000, 28, 28, 1))
test_features = test_features.reshape((10000, 28, 28, 1))

train_features = train_features.astype("float32")/255
test_features = test_features.astype("float32")/255

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from keras import layers
from keras import models
from keras import regularizers

def se(input):
    init = input
    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape((1, 128))(se)
    se = layers.Dense(128//32, activation='relu')(se)
    se = layers.Dense(128, activation='sigmoid')(se)
    #print(type(input))
    #x = tf.keras.layers.multiply([init, se])
    return se

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#se = models.clone_model(model)
#se = layers.Reshape((5,8))
model.add(layers.BatchNormalization())
model2 = se(model)

#model.add(layers.GlobalAveragePooling2D())
#model.add(layers.Reshape((1, 128)))
#model.add(layers.Dense(128//32, activation='relu'))
#model.add(layers.Dense(128, activation='sigmoid'))
#model.add(layers.Dense(128, activation='sigmoid'))
#model = tf.keras.layers.multiply([se, model])


model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(10, activation='softmax', use_bias=False, kernel_regularizer=regularizers.l1(0.00025)))

model2.summary()

#model.compile(optimizer ='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
#print(train_features.shape)
#model.fit(train_features, train_labels, epochs=20, batch_size=32)

#tfjs.converters.save_keras_model(model,'tfjsmodel')
#print(model.evaluate(test_features, test_labels)[1])