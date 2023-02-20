import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

ds = tfds.load('fashion_mnist', split='train')

#Visualise the data
#Use numpy to transform the data
import  numpy as np
ds.as_numpy_iterator().next()['label']
dataiterator = ds.as_numpy_iterator()

def scale_images(data):
    image = data['image']
    return image/255

fig, ax = plt.subplots(ncols=4, figsize =(20,20))
for i in range(4):
    pic  = dataiterator.next()
    ax[i].imshow(np.squeeze(pic['image']))
    ax[i].title.set_text(pic['label'])
plt.show()

# Build a data pipeline // Map // Cache // Shuffle // Batch // Prefetch
ds = ds.map(scale_images)
ds = ds.cache()
ds = ds.shuffle(50000)
ds = ds.batch(128)
ds = ds.prefetch(64)
