#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas.io.parsers import read_csv

# Es el cuadrado porque son matrices unidimensionales
def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y)**2
    return Aux.sum() / (2 * len(X)) # m = len(X)


def ecuacionNormal(X, Y):
    
    return (np.linalg.inv(np.dot(np.transpose(X),X))) * np.transpose(X) * Y


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


def gradiente (X,Y,Theta, alpha):
    m=np.shape(X)[0]
    H=np.dot(X,Theta)
    return Theta - (alpha/m) * np.dot(np.transpose(X),(H-Y))

def descenso_gradiente(X,Y,Theta):
    alpha = [0.1,0.3,0.01,0.03, 0.001, 0.003]
    ThetaOG = Theta
    for i in alpha:
        costes = np.zeros(1500)
        Theta = ThetaOG
        for a in range(1500):
            Theta=gradiente(X,Y,Theta,i)
            costes[a]=coste(X,Y,Theta)
        plt.xlabel("Numero de iteraciones")
        plt.ylabel(" theta")
        plt.plot(costes)
        plt.savefig("resultado{0}.png".format(i) )
        plt.clf() #limpiar

def normalize_mu_sigma(X):
    X_n = np.empty_like(X, dtype=float)
    n = np.shape(X)[1]
    mu = np.zeros(n)
    sigma = np.zeros(n)
    for i in range(n):
        mu[i] = np.mean(X[:,i])
        sigma[i] = np.std(X[:,i])
        X_n[:, i] = (X[:, i] - mu[i]) / sigma[i]
    #X_n = (X  - mu) / sigma
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
    descenso_gradiente(X,Y,Theta)

    ecuacionNormal(X,Y)

main()
# %%
