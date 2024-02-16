import numpy as np
pri = lambda arr : print(arr)

a = np.zeros((2, 3, 3), dtype='int32')
pri(a)
b = np.ones((4, 2, 2))
pri(b)
c = np.full_like(a.shape, 4, dtype='int64')
pri(c)
# random decimal numbers
d = np.random.rand(4, 2)
pri(d)
e = np.random.random_sample(a.shape)
pri(e)
# random integer values
f = np.random.randint(1, size=(3, 4))
pri(f)
g = np.identity(5)
pri(g)
arr1 = np.array([[1, 2, 3], [1, 2, 3]])
r1 = np.repeat(arr1, 3, axis=0)
pri(r1)

# build [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 9, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
# without actually typing it

output = np.ones((5, 5))
z = np.zeros((3, 3))
z[1,1] = 9
output[1:-1, 1:-1] = z
pri(output),

#### Be carefule when copying arrays!!!!
a = np.array([1, 3, 5, 6])
b = a.copy()
b[0] = 100
pri(b)
pri(a)