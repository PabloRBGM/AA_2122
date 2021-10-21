
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import polynomial
import scipy.optimize as opt
import math
import sklearn.preprocessing as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas.io.parsers import read_csv

# Calcula la funcion sigmoide 
def sigmoide(z):
    return 1/(1+math.e**(-z))

# 2.2 Calcula el coste (regularizado)
def coste(theta, X, Y ,_lambda):
    H = sigmoide(np.matmul(X,theta)) # dot?
    m = np.shape(X)[0]
    #esto funciona por el tamaño de las matrices. usamos la formula con las traspuestas
    #coste = (-1 / m) * (np.dot(Y, np.log(H)) + np.dot((1-Y),np.log(1 - H)))
    
    coste = (- 1 / (len(X))) * np.sum( Y * np.log(H) + (1 - Y) * np.log(1 - H) )

    regularizacion = (_lambda /(2+m) ) * np.sum(np.power(theta,2))
    coste += regularizacion
    return coste



# 2.2 Calcula el gradiente (regularizado)
def gradiente(Theta, X, Y, _lambda):
    H = sigmoide(np.matmul(X, Theta))
    m = np.shape(X)[0]

    return (1/m) * np.matmul(np.transpose(X), H - Y) + ((_lambda / m ) * Theta)

# #Devuelve el theta que buscamos
# def descenso_gradiente(X,Y,Theta):
#     costes = np.zeros(10)    
#     for a in range(10):
#         Theta=gradiente(Theta, X,Y)
#         costes[a]=coste(Theta, X,Y)   
#     return Theta




def plot_decisionboundary(X, Y, theta, poly):
    #plt.figure()
    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])#min y max de x1
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])#min y max de x2
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    #poly=sp.PolynomialFeatures(6)
    #
    #h = sigmoide(np.dot(np.c_[np.ones((np.ravel(xx1).shape[0], 1)), np.ravel(xx1), np.ravel(xx2 )],theta))
    # print(np.shape(aux))
    # print(np.shape(theta))
    h = sigmoide(np.dot(poly.fit_transform(np.c_[np.ravel(xx1),np.ravel(xx2 )]) ,theta))
    h = np.reshape(h,np.shape(xx1))
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.savefig("prueba.png")
    plt.close()


# def pinta_frontera(Theta, X, Y):
#     #plt.figure()
#     x1_min, x1_max = np.min(X[:, 1]), np.max(X[:, 1])#min y max de x1
#     x2_min, x2_max = np.min(X[:, 2]), np.max(X[:, 2])#min y max de x2

#     xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

#     #np.ones añade una columna de 1s
#     h = sigmoide(np.dot(np.c_[np.ones((np.ravel(xx1).shape[0], 1)),
#                  np.ravel(xx1), #vector columna
#                 np.ravel(xx2 )],Theta))
    
#     h = np.reshape(h, np.shape(xx1))

#     # el cuarto parámetro es el valor de z cuya frontera se quiere pintar
#     plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
#     plt.savefig("frontera.png")
#     plt.close()

def main():
    valores = read_csv("ex2data2.csv", header=None).to_numpy()
    X = valores[:, :-1]
    Y = valores[:, -1]
    n = np.shape(X)[1]  
    m = np.shape(X)[0]

    #apartado 2.1
    poly = sp.PolynomialFeatures(6).fit_transform(X)
    
    #print(np.shape(X)[1])


    #X=np.hstack([np.ones([m,1]), X])
    #print(X)
    Theta = np.zeros(np.shape(poly)[1])
    print(np.shape(poly)[1])
    print(np.shape(Theta))

    # 2.1 Grafica de los valores
    pos = np.where(Y == 1)
    plt.scatter(poly[pos, 1], poly[pos, 2], marker = '+', c='k', linewidths=2.0)
    pos = np.where(Y == 0)
    plt.scatter(poly[pos, 1], poly[pos, 2], marker = '.', c='green', linewidths=3.0)
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')

    #plt.legend('Admitted')
    
    #plt.show()  
    #print(coste(Theta, X, Y, 1000))
    #print(gradiente(Theta,X, Y, 1))
    _lamda = 1000
    result=opt.fmin_tnc(func=coste, x0 = Theta, fprime = gradiente, args=(poly, Y, _lamda))
    theta_opt = result[0]

    #print(np.shape(theta_opt)[0])

    plt.savefig("prueba.png")
    plot_decisionboundary(X,Y,theta_opt, poly)

   

main()