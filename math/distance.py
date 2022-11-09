import numpy as np


def ChebyshevDistance(x, y):
    x = np.array(x)
    print('x: ', x)
    y = np.array(y)
    print('y: ', y)
    return np.max(np.abs(x - y))


print('ChebyshevDistance：', ChebyshevDistance([1, 5], [4, 7]))


def StandardizedEuclideanDistance(x, y):
    x = np.array(x)
    y = np.array(y)
    
    X = np.vstack([x, y])
    sigma = np.var(X, axis=0, ddof=1)
    return np.sqrt(((x - y) ** 2 / sigma).sum())


print('StandardizedEuclideanDistance： ', StandardizedEuclideanDistance([1, 5], [4, 7]))
print('StandardizedEuclideanDistance： ', StandardizedEuclideanDistance(2, 4))