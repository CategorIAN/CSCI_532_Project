from functools import reduce
import numpy as np
import random
import pandas as pd

class MatrixProduct:
    def __init__(self, dimensionList):
        if self.valid(dimensionList):
            self.matDims = dimensionList
            self.n = len(dimensionList)
            self.table = np.vectorize(lambda n: int(n))(np.zeros((self.n, self.n)))
            self.dynamic = lambda state: self.table[state[0], state[1]]
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

    def deltaStates(self, state, action):
        (i, k, j) = (state[0], action, state[1])
        return {(i, k), (k + 1, j)}

    def actionValue(self, state):
        v = self.v(state)
        def f(action):
            return sum([self.dynamic(s) for s in self.deltaStates(state, action)]) + v(action)
        return f

    def optPair(self, state):
        if state[0] == state[1]:
            return (None, 0)
        else:
            valueFunc = self.actionValue(state)
            minPair = lambda av, bv: av if av[1] <= bv[1] else bv
            return reduce(minPair, [(a, valueFunc(a)) for a in self.actionSpace(state)])

    def optMults(self):
        for (i, j) in self.stateSpace():
            (_, minMults) = self.optPair((i, j))
            self.table[i, j] = minMults
        return self.dynamic((0, self.n - 1))

    #==============================================================================================================#

    def featVec(self, state):
        return np.array([state[0], state[1]])

    def netData(self, states):
        (iList, jList) = [list(t) for t in list(zip(*states))]
        return pd.DataFrame({"i": iList, "j": jList})

    def run(self, nn, D, eps, eta):
        (S, V) = (set(), {self.finalState})
        Q = set(random.sample(self.stateSpace, 5))|{self.finalState}
        def go(D, S, V, Q):
            if len(Q) == 0:
                return self.netData(set(random.sample(D, 5))|S)







