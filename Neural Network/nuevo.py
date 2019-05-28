
import numpy as np
import numpy.linalg as ln

classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def sigmoid(u):
  return 1. / (1. + np.e ** -u)

def back_propagation(X, labels, m, regularization=True):
  """Multilayer Neural Network
  input units:  4
  output units: 3
  :param X: d x n input matrix (d will be 4 for Iris)
  :param m: number of intermediate units
  """

  d, n = X.shape
  X = np.vstack((np.ones(n), X)).T # augumented; n x d+1

  # read label, and convert 3 unit format (001, 010, 100)
  b = -1 * np.ones((n, 3))
  for i in range(n):
    idx = classes.index(labels[i])
    b[i, idx] = 1.

  # weight matrix from input layer (d+1=3) to intermediate layer (m)
  W01 = np.random.randn(m, d+1)

  # weight matrix from intermediate layer (m) to output layer (3)
  W12 = np.random.randn(3, m)

  epoch = 0
  learning_rate = .01
  th = 1e-1
