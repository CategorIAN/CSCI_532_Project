import numpy as np
from functools import reduce
import random


class NeuralNet:
    def __init__(self, data):
        self.data = data

    def featVec(self, state):
        return np.array([state[0], state[1]])

    def targetvec(self):
        return lambda s: self.data[s]

    def randomWeights(self, rowDim, colDim):
        return np.random.rand(rowDim, colDim)

    def weightList(self, featNum, hidLayers, targetNum):
        appendWeight = lambda wL, dim: wL + [self.randomWeights(dim, wL[-1].shape[0])]
        dims = [featNum] + hidLayers + [targetNum]
        return reduce(appendWeight, dims[2:], [self.randomWeights(*reversed(dims[0:-1:]))])

    def stochastic_online_gd(self):
        vec_func = self.featVec  # create vector function for data
        base_index = random.sample(self.data.keys())  # create a shuffled index for iteration
                   # target length needs to be number of classes
        r = self.targetvec()