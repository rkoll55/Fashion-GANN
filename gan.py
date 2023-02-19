import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

ds = tfds.load('fashion_mnist', split='train')