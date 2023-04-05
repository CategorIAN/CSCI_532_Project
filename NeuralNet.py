import numpy as np
from functools import reduce
import random
import pandas as pd
from tail_recursive import tail_recursive


class NeuralNet:
    def __init__(self, data):
        self.data = data
        self.featVec = lambda state: np.array([state[0], state[1]])
        self.target = lambda s: self.data[s]

    def randomWeights(self, rowDim, colDim):
        return np.random.rand(rowDim, colDim)

    def weightList(self, featNum, hidLayers, targetNum):
        appendWeight = lambda wL, dim: wL + [self.randomWeights(dim, wL[-1].shape[0])]
        dims = [featNum] + hidLayers + [targetNum]
        return reduce(appendWeight, dims[2:], [self.randomWeights(*reversed(dims[0:2]))])

    def permute(self, index):
        return random.sample(index, len(index))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_v(self, x):
        return np.vectorize(self.sigmoid)(x)

    def dsigmoid_v(self, x):
        return 1 if x is None else x * (1 - x)

    def calc_Hidden(self, weights, x):
        appendHidden = lambda zs, w: zs + [self.sigmoid_v(w @ zs[-1])]
        return reduce(appendHidden, weights[:-1], [x])


    def epoch(self, eta, hidden_vector):
        @tail_recursive
        def online_update(index_remaining, ws, y_acc):
            if len(index_remaining) == 0:
                y_idx, y_values = zip(*y_acc)
                return (self.permute(index), ws, pd.Series(y_values, y_idx))
            else:
                s = index_remaining[0]
                x = self.featVec(s)
                zs = self.calc_Hidden(ws, x)
                v = (ws.iloc[-1] @ zs[-1])[0]
                error = np.array([self.target(s) - v])
                grads = []
                wzs = list(zip(ws, zs))
                previous_z = None
                for (w, z) in reversed(wzs):
                    grads = [np.outer(error * self.dsigmoid_v(previous_z), z)] + grads
                    error = error @ w
                    previous_z = z
                new_ws = pd.Series(zip(ws, grads)).map(lambda wg: wg[0] + eta * wg[1])
                return online_update.tail_call(index_remaining[1:], new_ws, y_acc + [(s, v)])
        start_ws = pd.Series(self.weightList(len(next(iter(self.data))), hidden_vector, 1))
        return online_update(self.permute(self.data.keys()), start_ws, [])



