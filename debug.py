
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random

br = 100
a = np.full([br*br],1)

np.eye(10,dtype="float32")[a]

np.eye(10)[np.random.randint(0,10, size=20)]


onehotdg = np.eye(10)[np.full([br*br],digit).reshape(-1)]
    