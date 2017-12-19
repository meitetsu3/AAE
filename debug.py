
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random

def clulster_face(num_l, sel):
    shift = 1.7
    r = (2.0 * np.pi / float(num_l)) * sel
    new_x = shift * np.cos(r)
    new_y = shift * np.sin(r)
    return np.array([new_x, new_y]).reshape((,2))

selector = np.array([0,1,2,3,4,5])

face = clulster_face(10,selector)

r = 2.0 * np.pi / float(10)*selector
cos(r)
        