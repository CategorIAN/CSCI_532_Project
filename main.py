from MatrixProduct import MatrixProduct
from NeuralNet import NeuralNet


def f(i):
    if i == 1:
        P = MatrixProduct([(2, 3), (3, 4), (4, 5)])
        print(P.optMults()) # I should get 64.
        print(P.optMults_NN(eta = .001, hidden_vector=[3]))
    if i == 2:
        P = MatrixProduct(20 * [(1, 1)])
        print(P.optMults())  # I should get 64.
        print(P.optMults_NN(eta=.001, hidden_vector=[3]))
    if i == 3:
        P = MatrixProduct([(2, 3), (3, 4), (4, 5)])
        print(P.stateSpace())



if __name__ == '__main__':
    f(1)

