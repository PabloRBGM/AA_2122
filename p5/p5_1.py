#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import scipy.optimize as opt
import math
import sklearn.preprocessing as sp
from pandas.io.parsers import read_csv
from scipy.io import loadmat

#ir caso a caso de X viendo que predice cada theta y quedarnos con el mayor y luego comprobar si es realmente ese
#por los 5000 casos, probar si es 0,1,2...9, ver cual es mayor y lo guardamos, luego comprobar si es cierto

# Calcula la funcion sigmoide 
def sigmoide(z):
    return 1/(1+math.e**(-z))

#  Calcula el coste (regularizado)
def costeReg(theta, X, Y , reg):
    H = np.dot(X, theta)
# Es el cuadrado porque son matrices unidimensionales
    Aux = np.power((H - Y),2)
    costeCero = Aux[:,1:]
    costeCero = np.hstack([np.zeros([np.shape(Aux)[0], 1]), costeCero]) 
    coste = (1/ (2 * np.shape(X)[0])) * np.sum(costeCero)

    Thetaceros = theta[:,1:]
    Thetaceros = np.hstack([np.zeros([np.shape(theta)[0], 1]), Thetaceros]) 
    regularizacion = (reg /(2*np.shape(X)[0]) ) * np.sum(np.power(Thetaceros,2))
    coste += regularizacion
    return coste

#  Calcula el gradiente (regularizado)

def gradiente(Theta, X, Y, _lambda):
    m=np.shape(X)[0]
    H=np.dot(X,Theta)

    aux =  (1/m) * np.dot(np.transpose(X),(H-Y))

    #quitar la primera columna de theta para calcular la regularizacion
    Thetaceros = Theta[:,1:]
    Thetaceros = np.hstack([np.zeros([np.shape(Theta)[0], 1]), Thetaceros]) 
    reg = ((_lambda/m) * Thetaceros)

    return aux + reg
    

def main():

    data = loadmat('ex5data1.mat')
    X=data["X"]
    Y=data["y"]
    Xval=data["Xval"]
    Yval=data["yval"]
    Xtest=data["Xtest"]
    Ytest=data["ytest"]

    _lambda = 1

    Theta = np.ones((1,2))
    print(costeReg(Theta,X,Y,_lambda))
    print(gradiente(Theta, X,Y,_lambda))
  

main()
# %%


# diez clasificadores, aplicar la regresion logistica 10 veces,
# en p2, las etiquetas son 0,1; la y del fichero va de 1 a 10
# tenemos que cambiar la y con n valores por una que solo tenga 0s y 1s
