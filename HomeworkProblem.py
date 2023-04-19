import numpy as np
from functools import reduce

class HomeworkProblem:
    state = ({2, 3}, 3)
    W1 = np.array([[1, 0, 1, 1, 0, 0, 2, 1], [1, 0, 0, 0, 0, 1, 1, 0]])
    W2 = np.array([[1.5, 2], [-1, 2]])
    U = np.array([3, 2])
    theta = [W1, W2, U]

    def sigmoid_v(self, x):
        return np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(x)

    def ourLambda(self, n):
        def f(state):
            P = state[0]
            x1 = np.vectorize(lambda v: int(v in P))(np.array(range(1, n + 1)))
            c = state[1]
            x2 = np.vectorize(lambda v: int(v == c))(np.array(range(1, n + 1)))
            return np.concatenate((x1, x2), axis = None)
        return f

    def neuralNetPred(self, n, theta):
        def f(state):
            def nn_layer(z, w):
                print(z)
                print(w @ z)
                return self.sigmoid_v(w @ z)
            final_z = reduce(nn_layer, theta[:-1], self.ourLambda(n)(state))
            return theta[-1] @ final_z
        return f

