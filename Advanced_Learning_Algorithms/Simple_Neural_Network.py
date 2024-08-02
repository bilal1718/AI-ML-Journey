import numpy as np


def sigmoid(z):
  return 1/(1 + np.exp(-z))

X=np.array([2,1])
y=np.array([0,1])
