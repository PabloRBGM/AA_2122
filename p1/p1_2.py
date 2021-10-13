#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas.io.parsers import read_csv

#Calcula el coste 
def coste(X, Y, Theta):
    H = np.dot(X, Theta)
# Es el cuadrado porque son matrices unidimensionales
    Aux = (H - Y)**2
    return Aux.sum() / (2 * len(X)) # m = len(X)

#Ecuacion Normal, es una formula que calcula theta. 
#Es menos eficaz cuantos mas parametros de entrenamiento y mas casos de prueba hay
def ecuacionNormal(X, Y):
    Aux=(np.linalg.inv(np.dot(np.transpose(X),X)))
    return  np.dot(np.dot(Aux, np.transpose(X)), Y)

#Calcula el gradiente con for en vez de forma vectorizada
# def gradiente_for(X, Y, Theta, alpha):
#     NuevaTheta = Theta
#     m = np.shape(X)[0]
#     n = np.shape(X)[1]
#     H = np.dot(X, Theta)
#     Aux = (H - Y)
#     for i in range(n):
#         Aux[i] = Aux * X[:, i]#toda la fila de la columna i
#         NuevaTheta[i] -= (alpha / m) * Aux[i].sum()
#     return NuevaTheta

#Calcula el gradiente de manera vectorizada
def gradiente (X,Y,Theta, alpha):
    m=np.shape(X)[0]
    H=np.dot(X,Theta)
    return Theta - (alpha/m) * np.dot(np.transpose(X),(H-Y))

#Devuelve el theta que buscamos
def descenso_gradiente(X,Y,Theta, alpha):
    costes = np.zeros(1500)    
    for a in range(1500):
        Theta=gradiente(X,Y,Theta,alpha)
        costes[a]=coste(X,Y,Theta)   
    return Theta

#Guarda en diferentes archivos la funcion de coste obtenida con 
#diferentes valores de alpha
def descenso_gradiente_archivos(X,Y,Theta):
    alpha = [0.1,0.3,0.01,0.03, 0.001, 0.003]
    ThetaOG = Theta
    for i in alpha:
        costes = np.zeros(1500)
        Theta = ThetaOG
        for a in range(1500):
            Theta=gradiente(X,Y,Theta,i)
            costes[a]=coste(X,Y,Theta)
        plt.xlabel("Numero de iteraciones")
        plt.ylabel("Coste de theta")
        plt.plot(costes)
        plt.savefig("resultado{0}.png".format(i) )
        plt.clf() #limpiar

#Normaliza los valores de entrenamiento
def normalize_mu_sigma(X):
    X_n = np.empty_like(X, dtype=float)
    n = np.shape(X)[1]
    mu = np.zeros(n)
    sigma = np.zeros(n)
    for i in range(n):
        mu[i] = np.mean(X[:,i])
        sigma[i] = np.std(X[:,i])
        X_n[:, i] = (X[:, i] - mu[i]) / sigma[i]
    return [X_n, mu, sigma]

def main(): 
    valores = read_csv("ex1data2.csv", header=None).to_numpy().astype(float)
    X_n, mu, sigma = normalize_mu_sigma(valores)
    X = X_n[:, :-1]
    Y = X_n[:, -1]
    m = np.shape(X_n)[0]#numero de casos de entrenamiento (filas)
    n = np.shape(X_n)[1]#numero de variables (columnas - la ultima)

    np.hstack([np.ones([m,1]), X])
    Theta = np.zeros(np.shape(X)[1])

    #descomentar para guardar los archivos
    #descenso_gradiente_archivos(X,Y,Theta)#para los archivos

    #1650 pies cuadrado y 3 habitaciones
    thetaGradiente=descenso_gradiente(X,Y,Theta,0.1)
    valor=thetaGradiente[0]*1650 + thetaGradiente[1]*3
    print(valor)

    thetaNormal=ecuacionNormal(X,Y)
    valor2=thetaNormal[0]*1650 + thetaNormal[1]*3
    print(valor2)

    print(valor - valor2)


main()
# %%
