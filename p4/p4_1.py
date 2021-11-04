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

#  Calcula el coste sin regularizar
def coste(theta1,theta2, X, Y ):
    a_1,a_2,H = forward_propagate(X, theta1, theta2)
    m = np.shape(X)[0]
    # usamos la formula con las traspuestas
    coste = (- 1 / (m)) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H + 1e-9))
    
    return coste

def coste_reg(theta1, theta2, X, Y, _lambda):
    a_1, a_2, H = forward_propagate(X, theta1, theta2)
    theta1 = theta1[:,1:]
    theta2 = theta2[:,1:]
    m = np.shape(X)[0]
    # usamos la formula con las traspuestas
    coste = (- 1 / (m)) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H + 1e-9))
    
    regularizacion = (_lambda /(2*m)) * (np.sum(np.power(theta1,2)) + np.sum(np.power(theta2,2)))
    coste += regularizacion
    
    return coste

#  Calcula el gradiente (regularizado)
def gradiente(Theta, X, Y, _lambda):
    H = sigmoide(np.matmul(X, Theta))
    m = np.shape(X)[0]
    return (1/m) * np.matmul(np.transpose(X), H - Y) + ((_lambda / m ) * Theta)

# Propagación
def forward_propagate(X, theta1, theta2):
    m = np.shape(X)[0]
    a_1 = np.hstack([np.ones([m, 1]), X]) 
    z_2 = np.matmul(a_1, np.transpose(theta1)) 
    a_2=sigmoide(z_2)
    a_2 = np.hstack([np.ones([m, 1]), a_2]) 
    z_3 = np.matmul(a_2, np.transpose(theta2)) 
    h=sigmoide(z_3)
    return a_1, a_2, h

def main():

    data = loadmat('ex4data1.mat')
    y = np.ravel(data[ 'y' ])
    X = data [ 'X' ]

    weigths = loadmat('ex4weights.mat')
    theta1 = weigths['Theta1'] # 25 x 401
    theta2 = weigths['Theta2'] # 10 x 26 

    m = len(y)
    num_labels = 10

    #Para tener mas ordenada la matriz y. Para evitar pasarnos con los indices
    y = (y-1)
    y_onehot = np.zeros((m,num_labels)) # 5000 x 10
    for i in range(m):
        y_onehot[i][y[i]] = 1

    print(coste(theta1, theta2, X, y_onehot))#coste sin regularizar
    print(coste_reg(theta1, theta2, X, y_onehot,1))#coste regularizado

main()
# %%


# diez clasificadores, aplicar la regresion logistica 10 veces,
# en p2, las etiquetas son 0,1; la y del fichero va de 1 a 10
# tenemos que cambiar la y con n valores por una que solo tenga 0s y 1s
