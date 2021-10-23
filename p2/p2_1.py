#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
from pandas.io.parsers import read_csv

# 1.2 Calcula la funcion sigmoide 
def sigmoide(z):
    return 1/(1+math.e**(-z))

# 1.3 Calcula el coste
def coste(theta, X, Y):
    H = sigmoide(np.matmul(X,theta)) # dot?
    coste = (-1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1-Y),np.log(1 - H)))
    #esto funciona por el tamaño de las matrices. podriamos usar la formula con las traspuestas
    return coste



# Calcula el gradiente
def gradiente(Theta, X, Y):
    H = sigmoide(np.matmul(X, Theta))
    return (1/len(X)) * np.matmul(np.transpose(X), H - Y)

# Pinta la frontera 
def pinta_frontera(Theta, X):

    x1_min, x1_max = np.min(X[:, 1]), np.max(X[:, 1])#min y max de x1
    x2_min, x2_max = np.min(X[:, 2]), np.max(X[:, 2])#min y max de x2

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    #np.ones añade una columna de 1s
    h = sigmoide(np.dot(np.c_[np.ones((np.ravel(xx1).shape[0], 1)),
                 np.ravel(xx1), #vector columna
                np.ravel(xx2 )],Theta))
    
    h = np.reshape(h, np.shape(xx1))

    # el cuarto parámetro es el valor de z cuya frontera se quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera.png")
    plt.close()

# Calcula el porcentaje de casos evaluados correctamente
def evaluacion_reg_logisitica(X, Y ,theta):  
    
    h = sigmoide(np.matmul(X,theta))

    #calcular aprobados
    apr_calculado = np.where(h >= 0.5)
    apr_real = np.where(Y == 1.0)
    apr_aux =np.in1d(apr_real, apr_calculado)
    apr_tmp = np.where(apr_aux == True)

    #calcular suspensos
    sus_calculado = np.where(h < 0.5)
    sus_real = np.where(Y != 1.0)
    sus_aux=np.in1d(sus_real, sus_calculado)
    sus_tmp = np.where(sus_aux == True)  

    total = np.shape(apr_tmp)[1] + np.shape(sus_tmp)[1]
    print(total/len(Y))



def main():
    valores = read_csv("ex2data1.csv", header=None).to_numpy()
    X = valores[:, :-1]
    Y = valores[:, -1]
    m = np.shape(X)[0]

    X=np.hstack([np.ones([m,1]), X])
    Theta = np.zeros(np.shape(X)[1])
    print(np.shape(X)[1])
    print(np.shape(Theta))

    # 1.1 Grafica de los valores
    pos = np.where(Y == 1)
    plt.scatter(X[pos, 1], X[pos, 2], marker = '+', c='k', linewidths=2.0)
    pos = np.where(Y == 0)
    plt.scatter(X[pos, 1], X[pos, 2], marker = '.', c='green', linewidths=3.0)
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')
    
    result=opt.fmin_tnc(func=coste, x0 = Theta, fprime = gradiente, args=(X, Y))
    
    theta_opt = result[0]
    pinta_frontera(theta_opt, X)   
    evaluacion_reg_logisitica(X, Y, theta_opt)


main()
# %%