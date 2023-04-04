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

    def deltaStates(self, state, scope):
        (i, j) = state
        def f(action):
            k = action
            return {(i, k), (k + 1, j)} if scope is None else {(i, k), (k + 1, j)}.intersection(scope)
        return lambda a: set() if i == j else f

    def actionValue(self, state, scope = None):
        v = self.v(state)
        def f(action):
            return sum([self.dynamic(s) for s in self.deltaStates(state, scope)(action)]) + v(action)
        return lambda a: 0 if state[0] == state[1] else f

    def optPair(self, state, scope = None):
        if state[0] == state[1]:
            return (None, 0)
        else:
            valueFunc = self.actionValue(state, scope)
            minPair = lambda av, bv: av if av[1] <= bv[1] else bv
            return reduce(minPair, [(a, valueFunc(a)) for a in self.actionSpace(state)])

    def optAction(self, state, scope = None):
        return self.optPair(state, scope)[0]

    def optValue(self, state, scope = None):
        return self.optPair(state, scope)[1]

    def optMults(self):
        for (i, j) in self.stateSpace:
            self.table[i, j] = self.optValue((i, j))
        return self.dynamic((0, self.n - 1))

    #==============================================================================================================#

    def featVec(self, state):
        return np.array([state[0], state[1]])

    def dataFrame(self, states):
        (iList, jList) = [list(t) for t in list(zip(*states))]
        return pd.DataFrame({"i": iList, "j": jList})

    def trainData(self, nn, D, eps, eta):
        (S, V, Q) = (set(), {self.finalState}, {self.finalState})
        def go(D, S, V, Q):
            if len(Q) == 0:
                return self.dataFrame(set(random.sample(D, 5))|S)
            else:
                state = Q[0]
                action = random.choice(self.actionSpace(state)) if random.random() < eps else self.optAction(state)







