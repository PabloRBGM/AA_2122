#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sklearn.preprocessing as sp

import math
from scipy.io import loadmat

# Calcula la funcion sigmoide 
def sigmoide(z):
    return 1/(1+math.e**(-z))

#  Calcula el coste (regularizado)
def costeReg(theta, X, Y , reg):
    H = np.dot(X, theta)
# Es el cuadrado porque son matrices unidimensionales
    Aux = np.power((H - Y),2)
    
    coste = (1/ (2 * np.shape(X)[0])) * np.sum(Aux)

    thetacopy = np.copy(theta)
    thetacopy[0] = 0

    regularizacion = (reg /(2*np.shape(X)[0]) ) * np.sum(np.power(thetacopy,2))
    coste += regularizacion
    return coste


#  Calcula el gradiente (regularizado)
def gradiente(Theta, X, Y, _lambda):
    m=np.shape(X)[0]
    H=np.dot(X,Theta)

    aux =  (1/m) * np.dot(np.transpose(X),(H-Y))

    #quitar la primera columna de theta para calcular la regularizacion
    thetacopy = np.copy(Theta)
    thetacopy[0] = 0
    reg = ((_lambda/m) * thetacopy)

    return aux + reg

def minFunc(theta, X, Y , reg):
    return (costeReg(theta, X, Y, reg), gradiente(theta, X, Y, reg))

def trainit(Theta, X, Y, reg, nIter):
    
    res = opt.minimize(
        fun=minFunc, 
        x0=Theta, 
        args=( X,Y,reg), 
        method='TNC',
        jac=True,
        options={'maxiter': nIter})
    return res

# Calcula el error (similar al coste pero sin regularizacion)
def error(theta, X, Y):
    H = np.dot(X, theta)
# Es el cuadrado porque son matrices unidimensionales
    Aux = np.power((H - Y),2)
    
    coste = (1/ (2 * np.shape(X)[0])) * np.sum(Aux)
    return coste


def polinomizar(X, p):
    return sp.PolynomialFeatures(p).fit_transform(X)[:,1:]

def normalizar(X):
    #valor - media / std
    xMean = np.mean(X, 0)
    xStd = np.std(X, 0)
    res = (X - xMean) / xStd
    return res, xMean, xStd

def normalizar_m_std(X, m, std):
    return (X - m) / std

# Pinta la grafica con la frontera de decision
def plot_decisionboundary(X, Y, theta, name):

    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])#min y max de x1
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])#min y max de x2
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    poly = sp.PolynomialFeatures(6)   
    
    h=costeReg(theta, X, Y, 0)
    h = np.reshape(h,np.shape(xx1))
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.savefig(name)
    plt.close()

def prueba(theta, xVal, n):
    aux = 0
    for i in range(n):
        aux += theta[i] * xVal[i]#**n
    return aux

def main():

    data = loadmat('ex5data1.mat')
    p = 8
    
    X=data["X"]
    Xpoly = polinomizar(X, p)
    Xnorm, Xmean, Xstd = normalizar(Xpoly)
    Xnorm = np.hstack([np.ones([np.shape(Xnorm)[0], 1]), Xnorm]) 

    Y=data["y"]
    y = np.ravel(Y)

    Xval=data["Xval"]
    Xval_poly = polinomizar(Xval, p)
    Xval_norm = normalizar_m_std(Xval_poly, Xmean, Xstd)
    Xval_norm = np.hstack([np.ones([np.shape(Xval_norm)[0], 1]), Xval_norm])

    Yval=data["yval"]
    Xtest=data["Xtest"]
    Ytest=data["ytest"] 
    _lambda = 100

    # Predicciones con sobreajuste
    Theta = np.ones(np.shape(Xnorm)[1])
    #res = trainit(Theta, Xnorm, y, _lambda, 500)
    #
    #plt.plot(X, Y, "x", c='red')
    #min_x = min(X)
    #max_X = max(X)
    #
    #range_X = np.arange(min_x, max_X, 0.05)
    #range_X = np.reshape(range_X, (-1,1))
    #norm_range_X = normalizar_m_std(polinomizar(range_X, p), Xmean, Xstd)
    #norm_range_X = np.hstack([np.ones([np.shape(norm_range_X)[0], 1]), norm_range_X]) 
    #newY = np.dot(norm_range_X, res.x)
    #
    #plt.plot(range_X, newY, c='blue')
    #plt.ylabel("Water flowing out of the dam (y)")
    #plt.xlabel("Change water level (x)")
    #plt.savefig("resultado3")

    # Curvas de aprendizaje
    errorCoste = np.zeros(np.shape(Xnorm)[0] )
    cValidationCoste = np.zeros(np.shape(Xnorm)[0] )
    for i in range(1, np.shape(Xnorm)[0] + 1):
        res = trainit(Theta, Xnorm[0:i], y[0:i], _lambda, 100)
      
        cValidationCoste[i - 1] = error(res.x, Xval_norm, np.ravel(Yval))
        errorCoste[i - 1] = error(res.x, Xnorm[0:i], y[0:i])
    
    plt.plot(errorCoste, c='blue')

    plt.ylabel("Error")
    plt.xlabel("Change water level (x)Number of training examples")
    plt.plot(cValidationCoste, c='orange')

    plt.savefig("Curvas aprendizaje 3 reg 100")

main()
# %%

