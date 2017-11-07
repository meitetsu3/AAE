# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:33:08 2017

@author: meite
"""

from keras.models import load_model
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

AAEencoder = load_model('keras_aae/encoder.h5')
AAEencoder.summary()

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

xtrain = xtrain.astype(np.float32) / 255.0
xtest = xtest.astype(np.float32) / 255.0

TestEncoded = AAEencoder.predict(xtrain) #training did not use any lable

# Plot
cm = plt.cm.get_cmap('tab10')
fig, ax = plt.subplots(1)

for i in range(10):
    y=TestEncoded[np.where(ytrain==i),1]
    x=TestEncoded[np.where(ytrain==i),0]
    color = cm(i)
    ax.scatter(x, y, label=str(i), alpha=0.9, facecolor=color, linewidth=0.15)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('2D latent code using AAE')

plt.show()