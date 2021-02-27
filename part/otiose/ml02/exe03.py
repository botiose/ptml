from matplotlib import pyplot as plt
import numpy as np

xlim = [0, 1]
ylim = [0, 1]

x = np.linspace(xlim[0], xlim[1], 100)
plt.ylim(ylim)

PlotCount = 4

yVal = np.exp(-1)
xVals = []

plt.plot([0, 1], [yVal, yVal], 'b--')

for l in range(1, PlotCount+1):
    y = np.exp(-x*l)
    plt.plot(x, y, label="$\lambda$: " + str(l))
    xVal = 1/l
    xVals.append(round(xVal, 2))
    plt.plot([xVal, xVal], [0, yVal], 'ro--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.annotate('$f_' + str(l) + '(x)=e^{-1}$', (1/l, np.exp(-1)),
                 textcoords="offset points",
                 xytext=(1,10));
    plt.legend()

plt.xticks(list(range(xlim[1]+1)) + xVals)
plt.yticks(list(range(ylim[1]+1)) + [round(yVal, 2)])
    
plt.show()
