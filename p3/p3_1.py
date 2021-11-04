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
def evaluacion(X,Y,thetaMat):
    # X (5000x401) thetaMat(t) (401x10)
    hMat = np.matmul(X,np.transpose(thetaMat))  # matriz de 5000x10 con las predicciones de cada theta
    nCases = np.shape(hMat)[0]
    MaxIndex = np.zeros(nCases)
    for i in range(nCases):
        MaxIndex[i]=np.argmax(hMat[i])
        if(MaxIndex[i]==0):
            MaxIndex[i]=10
    Num= np.sum(np.ravel(Y) == MaxIndex)
    Porcentaje=Num/nCases * 100
    print(Porcentaje)

def oneVsAll(X, y, num_etiquetas, reg):
    """
    oneVsAll entrena varios clasificadores por regresion logistica con termino de regularizacion
    'reg' y devuelve el resultado en una matriz, donde la fila i-esima correspone
    al clasificador de la etiqueta i-esima
    """
    porcentaje = np.zeros(10)
    theta_mat = np.zeros((num_etiquetas, np.shape(X)[1] + 1))
    m = np.shape(X)[0]
    X1s = np.hstack([np.ones([m, 1]), X])
    Theta = np.zeros(np.shape(X1s)[1])
    for i in range(num_etiquetas):
        #el numero que queremos buscar, 1s 2s 3s...
        y_i = (y==i) * 1
        if(i == 0):#el caso 0 son 10s, voltearlo
            y_i = (y==10) * 1
        y_i = np.ravel(y_i)
        print(np.shape(y_i))
        result = opt.fmin_tnc(func=coste, x0 = Theta, fprime = gradiente, args=(X1s, y_i, reg))
        theta_mat[i] = result[0]

    
    return theta_mat    

def main():
    #1.1 mostrar numeritos
    data = loadmat('ex3data1.mat')
# se pueden consultar las claves con data.keys( )
    y = data [ 'y' ]
    X = data [ 'X' ]
 # almacena los datos leídos en X, y
    sample = np.random.choice(X.shape[0],10)
    plt.imshow(np.transpose(np.reshape(X[sample, :],[-1,20])))
    plt.axis('off')
    #plt.show()

    m = np.shape(X)[0]
    X1s = np.hstack([np.ones([m, 1]), X])

    #1.2 Clasificacion de uno frente a todos    
    evaluacion(X1s,y,oneVsAll(X,y, 10, 0.1))

   
main()
# %%


# diez clasificadores, aplicar la regresion logistica 10 veces,
# en p2, las etiquetas son 0,1; la y del fichero va de 1 a 10
# tenemos que cambiar la y con n valores por una que solo tenga 0s y 1s
