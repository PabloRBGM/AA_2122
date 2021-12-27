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

def learning_curve(Theta, X, y, Xval, Yval, _lambda):
    m = np.shape(X)[0]
    errorCoste = np.zeros(m)
    cValidationCoste = np.zeros(m)
    for i in range(1, np.shape(X)[0] + 1):
        res = trainit(Theta, X[0:i], y[0:i], _lambda, 100)
      
        cValidationCoste[i - 1] = error(res.x, Xval, np.ravel(Yval))
        errorCoste[i - 1] = error(res.x, X[0:i], y[0:i])      

    #plt.plot(errorCoste, c='blue')
    #plt.ylabel("Error")
    #plt.xlabel("Change water level (x)Number of training examples")
    #plt.plot(cValidationCoste, c='orange')
    #
    #plt.savefig("Curvas aprendizaje 3 reg {0}.png".format(_lambda))

def lambdaSelection(_lambdas, Theta, X, y, Xval, Yval, Xtest, Ytest):
    n = np.size(_lambdas)
    coste = np.zeros(n)
    costeVal = np.zeros(n)
    costeTest = np.zeros(n)
    for i in range(n):
        res = trainit(Theta, X, y, _lambdas[i], 400)
        coste[i] = error(res.x, X, y)
        costeVal[i] = error(res.x, Xval, Yval)
        costeTest[i] = error(res.x, Xtest, Ytest)
    
    plt.plot(_lambdas, coste, c='blue')
    plt.plot(_lambdas, costeVal, c='orange')

    plt.ylabel("Error")
    plt.xlabel("lambda")
    #plt.plot(_lambdas, c='orange')

    plt.savefig("Seleccion del Lambda.png")
    return coste, costeVal, costeTest

def main():

    data = loadmat('ex5data1.mat')
    p = 8
    
    # Training
    X=data["X"]
    Xpoly = polinomizar(X, p)
    Xnorm, Xmean, Xstd = normalizar(Xpoly)
    Xnorm = np.hstack([np.ones([np.shape(Xnorm)[0], 1]), Xnorm]) 
    Y=data["y"]

    # Validation
    Xval=data["Xval"]
    Xval_poly = polinomizar(Xval, p)
    Xval_norm = normalizar_m_std(Xval_poly, Xmean, Xstd)
    Xval_norm = np.hstack([np.ones([np.shape(Xval_norm)[0], 1]), Xval_norm])
    Yval=data["yval"]

    # Test
    Xtest=data["Xtest"]
    Xtest_poly = polinomizar(Xtest, p)
    Xtest_norm = normalizar_m_std(Xtest_poly, Xmean, Xstd)
    Xtest_norm = np.hstack([np.ones([np.shape(Xtest_norm)[0], 1]), Xtest_norm])
    Ytest=data["ytest"] 

    Theta = np.ones(np.shape(Xnorm)[1])

    lambdas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    coste, costeVal, costeTest = lambdaSelection(lambdas, Theta, Xnorm, np.ravel(Y), Xval_norm, np.ravel(Yval), Xtest_norm, np.ravel(Ytest))
    print(costeTest[8])

main()
# %%

