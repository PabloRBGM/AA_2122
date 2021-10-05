#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas.io.parsers import read_csv

def coste(X, Y, th_0, th_1):
    m = len(X)
    sum = 0
    for i in range(m):
        sum += ((th_0 + th_1*X[i]) - Y[i])**2
    return sum / (2 * m)

def gradient(X, Y, alpha, th_0, th_1):
    m = len(X)
    for a in range(1500):
        sum_0 = sum_1 = 0
        for i in range(m):
            sum_0 += ((th_0 + th_1*X[i]) - Y[i])
            sum_1 += ((th_0 + th_1*X[i]) - Y[i]) * X[i]
        th_0 -= (alpha / m) * sum_0
        th_1 -= (alpha / m) * sum_1
        #print(coste(X, Y, th_0, th_1))
    plt.plot(X, Y, "x")
    min_x = min(X)
    max_X = max(X)
    min_y = th_0 + th_1 * min_x
    max_Y = th_0 +th_1 * max_X
    plt.plot([min_x, max_X], [min_y, max_Y])
    plt.savefig("resultado.pdf")

def make_data(t0_range, t1_range, X, Y):
    step = 0.1
    theta0 =np.arange(t0_range[0], t0_range[1],step)     
    theta1 =np.arange(t1_range[0], t1_range[1],step)
    theta0,theta1 = np.meshgrid(theta0, theta1)
    Coste = np.empty_like(theta0)
    #print(theta0.shape)
    for ix, iy in np.ndindex(theta0.shape):
        Coste[ix,iy] = coste(X,Y, theta0[ix,iy], theta1[ix,iy])
    return [theta0,theta1, Coste]

valores = read_csv("ex1data1.csv", header=None).to_numpy()
X = valores[:, 0]
Y = valores[:, 1]
gradient(X, Y, 0.01, 0, 0)

# surface
fig = plt.figure()
ax = fig.gca(projection = '3d') #Axes3D(fig)
data = make_data([-10, 10], [-1, 4], X, Y)
surf=ax.plot_surface(data[0], data[1], data[2], cmap=cm.coolwarm, linewidth=0,antialiased = False)

fig.colorbar(surf,shrink=0.5, aspect=5)
#primer parametro rotar en eje x, segundo rota en eje z 
ax.view_init(25,225)

# contour
# fig2 = plt.figure()


# fig2 = plt.contour(data[0], data[1], data[2], np.logspace(-2, 3, 20))
plt.show()
# %%
