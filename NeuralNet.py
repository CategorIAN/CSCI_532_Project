import numpy as np
from functools import reduce
import random
import pandas as pd
from tail_recursive import tail_recursive

class NeuralNet:

    def randomWeights(self, rowDim, colDim):
        return np.random.rand(rowDim, colDim)

    def weightList(self, featNum, hidLayers):
        appendWeight = lambda wL, dim: wL + [self.randomWeights(dim, wL[-1].shape[0])]
        dims = [featNum] + hidLayers + [1]
        return reduce(appendWeight, dims[2:], [self.randomWeights(*reversed(dims[0:2]))])

    def permute(self, index):
        return random.sample(index, len(index))

    def sigmoid_v(self, x):
        return np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(x)

    def dsigmoid_v(self, x):
        return 1 if x is None else x * (1 - x)

    def calc_Hidden(self, weights, x):
        appendHidden = lambda zs, w: zs + [self.sigmoid_v(w @ zs[-1])]
        return reduce(appendHidden, weights[:-1], [x])

    def epoch(self, data, eta, hidden_vector = None, start_ws = None):
        @tail_recursive
        def online_update(index_remaining, ws, y_acc):
            if len(index_remaining) == 0:
                return (ws, pd.Series(*zip(*y_acc)))
            else:
                s = index_remaining[0]
                zs = self.calc_Hidden(ws, np.array(s))
                v = (ws.iloc[-1] @ zs[-1])[0]
                def appendGrad(grad_err_z, w_z):
                    newGrad = [np.outer(grad_err_z[1] * self.dsigmoid_v(grad_err_z[2]), w_z[1])] + grad_err_z[0]
                    newError = grad_err_z[1] @ w_z[0]
                    return (newGrad, newError, w_z[1])
                grads = reduce(appendGrad, reversed(list(zip(ws, zs))), ([], np.array([data[s] - v]), None))[0]
                new_ws = pd.Series(zip(ws, grads)).map(lambda wg: wg[0] + eta * wg[1])
                return online_update.tail_call(index_remaining[1:], new_ws, y_acc + [(v, s)])

        start_ws = pd.Series(self.weightList(len(next(iter(data))), hidden_vector)) if start_ws is None else start_ws
        return online_update(self.permute(data.keys()), start_ws, [])



