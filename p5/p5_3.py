#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
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

def main():

    data = loadmat('ex5data1.mat')
    X=data["X"]
    X_1s = np.hstack([np.ones([np.shape(X)[0], 1]), X]) 
    Y=data["y"]
    Xval=data["Xval"]
    Xval_1s = np.hstack([np.ones([np.shape(Xval)[0], 1]), Xval]) 
    Yval=data["yval"]
    Xtest=data["Xtest"]
    Ytest=data["ytest"]

    _lambda = 0
    y = np.ravel(Y)
    Theta = np.array([1,1])
    cValidationCoste = np.zeros(np.shape(X)[0] )
    errorCoste = np.zeros(np.shape(X)[0] )
    for i in range(1, np.shape(X)[0] + 1):
        res = trainit(Theta, X_1s[0:i], y[0:i], _lambda, 70)
        cValidationCoste[i - 1] = error(res.x, Xval_1s, np.ravel(Yval))
        errorCoste[i - 1] = error(res.x, X_1s[0:i], y[0:i])


 
    plt.plot(errorCoste, c='blue')

    plt.ylabel("Error")
    plt.xlabel("Change water level (x)Number of training examples")
    plt.plot(cValidationCoste, c='orange')

    plt.savefig("Resultado Error Training-Cross")
    
    #print(costeReg(Theta,X_1s,y,_lambda))
    #print(gradiente(Theta, X_1s,y,_lambda))

main()
# %%

