from MatrixProduct import MatrixProduct


def f(i):
    if i == 1:
        P = MatrixProduct([(2, 4), (3, 4), (4, 5)])
        print(P.optMults()) # I should get 64.


if __name__ == '__main__':
    f(1)

