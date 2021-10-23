#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
import sklearn.preprocessing as sp
from pandas.io.parsers import read_csv

# Calcula la funcion sigmoide 
def sigmoide(z):
    return 1/(1+math.e**(-z))

# 2.2 Calcula el coste (regularizado)
def coste(theta, X, Y ,_lambda):
    H = sigmoide(np.matmul(X,theta)) 
    m = np.shape(X)[0]
    # usamos la formula con las traspuestas
    coste = (- 1 / (len(X))) * np.sum( Y * np.log(H) + (1 - Y) * np.log(1 - H) )

    regularizacion = (_lambda /(2+m) ) * np.sum(np.power(theta,2))
    coste += regularizacion
    return coste

# 2.2 Calcula el gradiente (regularizado)
def gradiente(Theta, X, Y, _lambda):
    H = sigmoide(np.matmul(X, Theta))
    m = np.shape(X)[0]

    return (1/m) * np.matmul(np.transpose(X), H - Y) + ((_lambda / m ) * Theta)

# Pinta la grafica con la frontera de decision
def plot_decisionboundary(X, Y, theta, name):

    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])#min y max de x1
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])#min y max de x2
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    poly = sp.PolynomialFeatures(6)   
    
    h=sigmoide(poly.fit_transform(np.c_[xx1.ravel(),xx2.ravel()]).dot(theta))
    h = np.reshape(h,np.shape(xx1))
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.savefig(name)
    plt.close()


def main():
    valores = read_csv("ex2data2.csv", header=None).to_numpy()
    X = valores[:, :-1]
    Y = valores[:, -1]
    n = np.shape(X)[1]
    m = np.shape(X)[0]

    # 2.1 Mapeo de Atributos
    poly = sp.PolynomialFeatures(6).fit_transform(X)    
    Theta = np.zeros(np.shape(poly)[1])

    # 2.0 Grafica de los valores
    pos = np.where(Y == 1)
    plt.scatter(poly[pos, 1], poly[pos, 2], marker = '+', c='k', linewidths=2.0)
    pos = np.where(Y == 0)
    plt.scatter(poly[pos, 1], poly[pos, 2], marker = '.', c='green', linewidths=3.0)
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')    
    
    # 2.3 Calculo de valor optimo
    _lamda = 150
    result=opt.fmin_tnc(func=coste, x0 = Theta, fprime = gradiente, args=(poly, Y, _lamda))
    theta_opt = result[0]

    name = "Regularizacion{0}.png".format(_lamda)
    #plt.savefig(name) # guarda el archivo sin frontera de decision
    plot_decisionboundary(X,Y,theta_opt, name)

   
main()
# %%