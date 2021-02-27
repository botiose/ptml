from matplotlib import pyplot as plt
import numpy as np

xlim = [-1, 2]
ylim = [0, 5]

x = np.linspace(xlim[0], xlim[1], 100)
plt.ylim(ylim)

PlotCount = 4

for l in range(1, PlotCount+1):
    y = np.exp(-x*l)
    plt.subplot(221 + l - 1)
    plt.plot(x, y, label="$\lambda$: " + str(l))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.show()
