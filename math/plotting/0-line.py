import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 11, 0.1)
y = x ** 3

plt.plot(x, y, 'r')
plt.xlim(0, 10)
plt.ylim(0, 1000)

plt.show()