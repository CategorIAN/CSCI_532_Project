import numpy as np
from functools import reduce
import random
import pandas as pd
from tail_recursive import tail_recursive


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


    def online_update(self, vec_func, r, eta, index):
        '''
        @param index_remaining: index left to iterate through
        @param w: the current weight matrix
        @param y_acc: the current set of accumulated predictions
        @return: new index, final weight matrix, and complete set of predictions after iterated through index
        '''

        @tail_recursive
        def f(index_remaining, ws, y_acc):
            if len(index_remaining) == 0:  # if there is nothing more to iterate through
                y_idx, y_values = zip(*y_acc)  # unzip to get y index and y values
                return (self.permute(index), ws, pd.Series(y_values, y_idx))
            else:
                s = index_remaining[0]  # the next index value
                x = vec_func(s)  # the next sample vector
                zs = self.calc_Hidden(ws, x)                   # the input and hidden layers
                v = (ws.iloc[-1] @ zs[-1])[0]                                # return a real value
                error = np.array([r(s) - v])                                     # return errors at each of the outputs
                grads = []
                wzs = list(zip(ws, zs))
                previous_z = None
                for (w, z) in reversed(wzs):
                    grads = [np.outer(error * self.dsigmoid_v(previous_z), z)] + grads   # create gradient
                    error = error @ w                                               # back propagate error
                    previous_z = z
                new_ws = pd.Series(zip(ws, grads)).map(lambda wg: wg[0] + eta * wg[1])           #calculate new weights                                     #calculate new gradients
                return f.tail_call(index_remaining[1:], new_ws, y_acc + [(s, v)])
        return f

    def stochastic_online_gd(self):
        vec_func = self.featVec  # create vector function for data
        base_index = random.sample(self.data.keys())  # create a shuffled index for iteration
                   # target length needs to be number of classes
        r = self.targetvec()
        def f(eta, hidden_vector):
            nrows = len(next(iter(self.data)))
            ws_init = pd.Series(self.weightList(nrows, hidden_vector, 1))  # initial randomized weights
            '''
            @param index: the index to iterate through
            @param start_w: the starting weight matrix to use for the epoch
            @return: new permuted index, weight matrix learned from data, and a series of predicted values
            '''

            def epoch(index, start_w):
                return self.online_update(vec_func, r, eta, index)(index, start_w, [])