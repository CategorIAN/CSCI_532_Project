import numpy as np
from functools import reduce


class NeuralNet:
    def randomWeights(self, rowDim, colDim):
        return np.random.rand(rowDim, colDim)

    def weightList(self, featNum, hidLayers, targetNum):
        appendWeight = lambda wL, dim: wL + [self.randomWeights(dim, wL[-1].shape[0])]
        dims = [featNum] + hidLayers + [targetNum]
        return reduce(appendWeight, dims[2:], [self.randomWeights(*reversed(dims[0:-1:]))])