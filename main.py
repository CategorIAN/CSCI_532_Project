from MatrixProduct import MatrixProduct
from NeuralNet import NeuralNet


def f(i):
    if i == 1:
        P = MatrixProduct([(2, 3), (3, 4), (4, 5)])
        print(P.optMults()) # I should get 64.
    if i == 2:
        N = NeuralNet()
        print(N.weightList(2, [3], 1))
    if i == 3:
        P = MatrixProduct([(2, 3), (3, 4), (4, 5)])
        print(P.stateSpace())


if __name__ == '__main__':
    f(1)

