import tensorflow as tf
import numpy as np

x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))

print(x1.shape)