import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU, Reshape,UpSampling2D

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
print(ds.as_numpy_iterator().next().shape) # should be 128 images, 28x28 with 1 for grayscale 

# sequential API for generator and discriminator
def build_generator():
    model = Sequential()
    # sequential used to build a stack of layers
    model.add(Dense(7*7*128, input_dim=128))
    # 128 input values mapped to high dimentional space. Makes 7*7*128 feature map
    model.add(LeakyReLU(0.2))
    # Helps identify non linearities in the dataset and smooths them out with sigmoid like function.  
    model.add(Reshape((7,7,128)))
    # takes the dense layer and generates the 7*7 image with 128 layer


    model.add(UpSampling2D())
    #Upscales to 14*14*128
    model.add(Conv2D(128,5,padding ='same'))
    #Extracts local features from the input data
    model.add(LeakyReLU(0.2))
    #Second leaky relu to help with non linearities

    model.add(UpSampling2D())
    model.add(Conv2D(128,5,padding ='same')) # gets 128 different features from the input data
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128,5,padding ='same'))
    model.add(LeakyReLU(0.2))
    # adds more params to the neural network to help it learn more
    model.add(Conv2D(128,5,padding ='same'))
    model.add(LeakyReLU(0.2))
    #should return 28*28*128
    model.add(Conv2D(1,4,padding='same',activation='sigmoid'))
    return model 