import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(x, 0)


x = np.array([-2, -1, 0, 1, 2, 3])
y = relu(x)
plt.plot(x, y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Rectified Linear Unit')
plt.show()
