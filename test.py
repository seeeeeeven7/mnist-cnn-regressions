import numpy as np


a = np.random.rand(4, 2)
b = np.random.rand(2)

print(a)
print(b)
print(a + b)

A = np.arange(8).reshape((2,4))
print(np.flip(np.flip(A, 0), 1))