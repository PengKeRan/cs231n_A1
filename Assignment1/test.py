import numpy as np

a = np.random.randint(10, size=3)
b = np.random.randint(10, size=[3, 4])
c = np.random.randint(10, size=[5, 5, 2])
c = c.reshape(-1)
print(c.shape)

# print(np.sqrt(np.sum(np.square(c[:, ]))))
