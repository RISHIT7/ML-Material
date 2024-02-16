import numpy as np

arr = np.array([[1, 2, 4, 5, 7], [3, 5, 6, 7, 8]])
print(np.min(arr))
print(np.max(arr))
print(np.min(arr, axis=1))
print(np.sum(arr))

#### Reorganizing arrays
before = np.array([[1, 3, 5, 6], [6, 4, 6, 8]])
after = before.reshape((2, 2, 2))
print(after)

#### Vertical stacking vectors
v1 = np.array([1, 2, 4, 6])
v2 = np.array([4, 5, 7, 8])

print(np.vstack([v1, v2, v2]))
#Horizontal stack
h1 = np.array([1, 2, 4, 6])
h2 = np.array([4, 5, 7, 8])

print(np.hstack((h1, h2)))