from keras.datasets import mnist
import tensorflowjs as tfjs
import matplotlib.pyplot as plt

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

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer ='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
print(train_features.shape)
model.fit(train_features, train_labels, epochs=20, batch_size=32)

tfjs.converters.save_keras_model(model,'tfjsmodel')
print(model.evaluate(test_features, test_labels)[1])