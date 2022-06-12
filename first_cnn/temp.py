from tensorflow.keras.datasets import mnist
import tensorflowjs as tfjs
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()

print(train_features.shape)
print(test_features.shape)

plt.imshow(train_features[3,:,:], cmap =plt.cm.binary)
plt.show()