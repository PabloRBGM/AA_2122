#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat


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

def main():

    data = loadmat('ex5data1.mat')
    X=data["X"]
    X_1s = np.hstack([np.ones([np.shape(X)[0], 1]), X]) 
    Y=data["y"]
   
    _lambda = 0
    y = np.ravel(Y)
    Theta = np.array([1,1])
    res = trainit(Theta, X_1s, y, _lambda, 70)
    print(res)

    plt.plot(X, Y, "x", c='red')
    min_x = min(X)
    max_X = max(X)
    min_y = res.x[0] + res.x[1] * min_x
    max_Y = res.x[0] + res.x[1] * max_X
    plt.plot([min_x, max_X], [min_y, max_Y], c='blue')
    plt.ylabel("Water flowing out of the dam (y)")
    plt.xlabel("Change water level (x)")
    plt.savefig("resultado1.png")
    
    
    print(costeReg(Theta,X_1s,y,_lambda))
    print(gradiente(Theta, X_1s,y,_lambda))

main()
# %%

