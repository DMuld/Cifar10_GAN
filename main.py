from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
# from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

# Rescale the data from 0 to 255 -> 0 to 1 NORMALIZE
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

inputs = keras.Input(shape=(32,32,3))

##################ENCODE##################

d1 = keras.layers.Conv2D(128, (3,3), padding='same')(inputs)
d2 = keras.layers.BatchNormalization()(d1)
d3 = keras.layers.Activation('relu')(d2)
d4 = keras.layers.MaxPooling2D((2,2), padding='same')(d3)
d5 = keras.layers.Conv2D(64, (3,3), padding='same')(d4)
d6 = keras.layers.BatchNormalization()(d5)
d7 = keras.layers.Activation('relu')(d6)
d8 = keras.layers.MaxPooling2D((2,2), padding='same')(d7)
d9 = keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(d8)
d10 = keras.layers.BatchNormalization()(d9)
d11 = keras.layers.Activation('relu')(d10)
encoded = keras.layers.MaxPooling2D((2,2), padding='same')(d11)

##################DECODE##################

d1 = keras.layers.Conv2D(16, (3,3), padding='same')(encoded)
d2 = keras.layers.BatchNormalization()(d1)
d3 = keras.layers.Activation('relu')(d2)
d4 = keras.layers.UpSampling2D((2,2))(d3)
d5 = keras.layers.Conv2D(64, (3,3), padding='same')(d4)
d6 = keras.layers.BatchNormalization()(d5)
d7 = keras.layers.Activation('relu')(d6)
d8 = keras.layers.UpSampling2D((2,2))(d7)
d9 = keras.layers.Conv2D(128, (3,3), padding='same')(d8)
d10 = keras.layers.BatchNormalization()(d9)
d11 = keras.layers.Activation('relu')(d10)
d12 = keras.layers.UpSampling2D((2,2))(d11)
d13 = keras.layers.Conv2D(3, (3,3), padding='same')(d12)
d14 = keras.layers.BatchNormalization()(d13)
decoded = keras.layers.Activation('sigmoid')(d14)

##################MODEL##################

model = keras.Model(inputs=inputs, outputs=decoded)

# Shows a summary of what the model will look like
model.summary()

# Compiles the training alg.
model.compile(optimizer='adam', loss='mae', metrics=['accuracy']) 

# Fits the training model
fittedModel = model.fit(x_train, x_train, batch_size=64, epochs=10, validation_data=(x_test, x_test))

# Predict the model
pred = model.predict(x_test)

##################PRINT IMAGES##################
# Make the graph.
# Shows 6 Origin, 6 Remade.

number = 12
half = number/2
sbplt, img = plt.subplots(1, number, figsize=(32,32))
for i in range(number):
    if (i < (half)):
        sample = x_test[i]
        img[i].imshow(sample, cmap='gray')
        img[i].set_title("Original: {}".format(i), fontsize=10)
        img[i].get_xaxis().set_visible(False)
        img[i].get_yaxis().set_visible(False)
    else:
        sample = pred[int(i-half)]
        img[i].imshow(sample, cmap='gray')
        img[i].set_title("Remade: {}".format((i-int(half))), fontsize=10)
        img[i].get_xaxis().set_visible(False)
        img[i].get_yaxis().set_visible(False)
plt.show()
