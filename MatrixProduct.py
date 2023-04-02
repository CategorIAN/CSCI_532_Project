from functools import reduce
import numpy as np

class MatrixProduct:
    def __init__(self, dimensionList):
        if self.valid(dimensionList):
            self.matDims = dimensionList
            self.n = len(dimensionList)
        else:
            raise Exception("Dimensions do not match.")

    def valid(self, dimensionList):
        def go(p, dimList):
            if len(dimList) == 0:
                return True
            else:
                q = dimList[0]
                if p[1] != q[0]:
                    return False
                else:
                    return go(q, dimList[1:])
        return go(dimensionList[0], dimensionList[1:]) if len(dimensionList) > 0 else False

    def optMults(self):
        multArray = np.vectorize(lambda n: int(n))(np.zeros((self.n, self.n)))
        for d in range(1, self.n):
            for i in range(self.n - d):
                j = i + d
                minMults = multArray[i + 1, j] + self.matDims[i][0] * self.matDims[i][1] * self.matDims[j][1]
                for k in range(i + 1, j):
                    mults = multArray[i, k] + multArray[k + 1, j] + self.matDims[i][0] * self.matDims[k][1] * self.matDims[j][1]
                    if mults < minMults:
                        minMults = mults
                multArray[i, j] = minMults
        return multArray[0, self.n - 1]




