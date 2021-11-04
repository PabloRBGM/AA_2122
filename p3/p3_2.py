#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import scipy.optimize as opt
import math
import sklearn.preprocessing as sp
from pandas.io.parsers import read_csv
from scipy.io import loadmat

# Calcula la funcion sigmoide 
def sigmoide(z):
    return 1/(1+math.e**(-z))

#  Calcula el coste (regularizado)
def coste(theta, X, Y ,_lambda):
    H = sigmoide(np.matmul(X,theta)) 
    m = np.shape(X)[0]
    # usamos la formula con las traspuestas
    coste = (- 1 / (len(X))) * np.sum( Y * np.log(H) + (1 - Y) * np.log(1 - H + 1e-6) )

    regularizacion = (_lambda /(2+m) ) * np.sum(np.power(theta,2))
    coste += regularizacion
    return coste

#  Calcula el gradiente (regularizado)
def gradiente(Theta, X, Y, _lambda):
    H = sigmoide(np.matmul(X, Theta))
    m = np.shape(X)[0]
    return (1/m) * np.matmul(np.transpose(X), H - Y) + ((_lambda / m ) * Theta)

# Calcula el porcentaje de éxito de nuestro modelo multi-clase
def evaluacion(y, a_3):
    MaxIndex=np.zeros(np.shape(a_3)[0])

    for i in range(np.shape(a_3)[0]):
        MaxIndex[i]=np.argmax(a_3[i]) + 1#mas uno por como esta ordenado,el 0 equivale a 1... el 9 a un 10
        
    Num= np.sum(np.ravel(y) == MaxIndex)
    Porcentaje=Num/np.shape(y)[0] * 100
    print(Porcentaje)

def main():
    data = loadmat('ex3data1.mat')
    y = data [ 'y' ] # 5000, 1
    X = data [ 'X' ] # 5000, 400

    weigths = loadmat('ex3weights.mat')
    theta1 = weigths['Theta1'] # 25 x 401
    theta2 = weigths['Theta2'] # 10 x 26 

    a_1 = np.hstack([np.ones([np.shape(X)[0], 1]), X]) # 5000 x 401
    z_2 = np.matmul(a_1, np.transpose(theta1)) # 5000 x 25
    a_2=sigmoide(z_2)
    a_2 = np.hstack([np.ones([np.shape(a_2)[0], 1]), a_2]) # 5000 x 26
    z_3 = np.matmul(a_2, np.transpose(theta2)) # 5000 x 10
    a_3=sigmoide(z_3)

    evaluacion(y, a_3)

main()
# %%