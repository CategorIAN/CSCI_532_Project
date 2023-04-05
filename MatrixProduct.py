from functools import reduce
import numpy as np
import random
import pandas as pd
from NeuralNet import NeuralNet
from tail_recursive import tail_recursive

class MatrixProduct:
    def __init__(self, dimensionList):
        if self.valid(dimensionList):
            self.matDims = dimensionList
            self.n = len(dimensionList)
            self.table = np.vectorize(lambda n: int(n))(np.zeros((self.n, self.n)))
            self.dynamicStd = lambda state: self.table[state[0], state[1]]
            self.stateSpace = self.getStateSpace()
            self.finalState = (0, self.n - 1)
        else:
            raise Exception("Dimensions do not match.")

    def valid(self, dimensionList):
        def go(p, dimList):
            if len(dimList) == 0:
                return True
            else:
                q = dimList[0]
                return (p[1] == q[0]) and go(q, dimList[1:])
        return go(dimensionList[0], dimensionList[1:]) if len(dimensionList) > 0 else False

    def getStateSpace(self):
        def dSpace(d):
            return [(i, i + d) for i in range(self.n - d)]
        return reduce(lambda l1, l2: l1 + l2, [dSpace(d) for d in range(self.n)])

    def actionSpace(self, state):
        return list(range(*state))

    def v(self, state):
        (i, j) = state
        def f(action):
            k = action
            return self.matDims[i][0] * self.matDims[k][1] * self.matDims[j][1]
        return f

    def deltaStates(self, state, D = None):
        (i, j) = state
        def f(action):
            if i == j:
                return set()
            else:
                k = action
                return {(i, k), (k + 1, j)} if D is None else {(i, k), (k + 1, j)}.intersection(D)
        return f

    def actionValue(self, state, D = None):
        dynamic = self.dynamicStd if D is None else lambda s: D[s]
        v = self.v(state)
        def f(action):
            if state[0] == state[1]:
                return 0
            else:
                return sum([dynamic(s) for s in self.deltaStates(state, D)(action)]) + v(action)
        return f

    def optPair(self, state, D = None):
        if state[0] == state[1]:
            return (None, 0)
        else:
            valueFunc = self.actionValue(state, D)
            minPair = lambda av, bv: av if av[1] <= bv[1] else bv
            return reduce(minPair, [(a, valueFunc(a)) for a in self.actionSpace(state)])

    def optAction(self, state, D = None):
        return self.optPair(state, D)[0]

    def optValue(self, state, D = None):
        return self.optPair(state, D)[1]

    def optMults(self):
        for (i, j) in self.stateSpace:
            self.table[i, j] = self.optValue((i, j))
        return self.dynamicStd((0, self.n - 1))

    #==============================================================================================================#

    def filter(self, condition, states):
        states_filtered = [[s] if condition(s) else [] for s in states]
        return reduce(lambda l1, l2: l1 + l2, states_filtered, [])

    def getTrainingData(self, D, eps):
        @tail_recursive
        def go(S, V, Q):
            if len(Q) == 0:
                return dict([(s, self.optValue(s, D)) for s in set(random.sample(D.keys(), min(len(D), 5)))|S])
            else:
                state = Q[0]
                possibleActions = self.actionSpace(state)
                if len(possibleActions) == 0:
                    newStates = []
                else:
                    if random.random() < eps:
                        action = random.choice(possibleActions)
                    else:
                        action = self.optAction(state, D)
                    newStates = self.filter(lambda s: s not in V, self.deltaStates(state)(action))
                return go.tail_call(S|{state}, V|set(newStates), Q[1:] + newStates)
        return go(set(), {self.finalState}, [self.finalState])


    def optMults_NN(self, eta, hidden_vector):
        NN = NeuralNet()
        @tail_recursive
        def go(t, D, eps,y, ws):
            print("-------------------------------------------------------------")
            print("t: {}".format(t))
            print("Predicted value from nn is {}.".format(None if y is None else y[self.finalState]))
            print("Predicted value from D is {}.".format(None if D == {} else D[self.finalState]))
            if t == 100000:
                return y[self.finalState]
            else:
                trainingData = self.getTrainingData(D, eps)
                #print("Training Data: {}".format(sorted(list(trainingData.keys()))))
                (newWs, newY) = NN.epoch(trainingData, eta, hidden_vector, ws)
                #print("Total Data: {}".format(sorted(list((D|trainingData).keys()))))
                return go.tail_call(t + 1, D|trainingData, max(0.995 * eps, 0.05), newY, newWs)
        return go(0, {}, 1, None, None)









