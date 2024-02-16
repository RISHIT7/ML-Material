import numpy as np

a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
print(a[1, 5]) # similar to a[1][1]
# get a specific row
print(a[1, :])
# Getting a little more fancy [startindex:endindex:stepsize]
print(a[0, 1:6:2])

a[1, 5] = 20
print(a)