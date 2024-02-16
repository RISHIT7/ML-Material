import numpy as np

a = np.array([1, 3, 5, 6, 7])
b = a + 2
c = np.sin(a)
print(b)
print(a + b)
print(c)

# Linear Algebra
a_ = np.full((2, 3), 1)
b_ = np.full((3, 2), 2)
c_ = np.matmul(a_, b_)
print(np.linalg.det(c_))
print(c_)

a = np.ones((5, 5))
b = np.zeros((3, 3))
b[1][1] = 9
a[1:4, 1:4] = b
c = np.array([[1, 2, 34, 5, 5], [2, 3, 4, 5, 6], [1, 5, 6, 7, 7], [4, 3, 5, 2, 6], [ 45, 6, 3, 2, 6]])
d = np.matmul(c, a)
print(f"{a} * {c} = {d}")
print(f"det(d) = {np.linalg.det(d)}")