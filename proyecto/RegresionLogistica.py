import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
import sklearn.preprocessing as sp
from sklearn.metrics import accuracy_score

# Calcula la funcion sigmoide 
def sigmoide(z):
    return 1/(1+math.e**(-z))

# 2.2 Calcula el coste (regularizado)
def coste(theta, X, Y ,_lambda):
    H = sigmoide(np.matmul(X,theta)) 
    m = np.shape(X)[0]
    # usamos la formula con las traspuestas
    coste = (- 1 / (len(X))) * np.sum( Y * np.log(H) + (1 - Y) * np.log(1 - H) )

    regularizacion = (_lambda /(2+m) ) * np.sum(np.power(theta,2))
    coste += regularizacion
    return coste

# 2.2 Calcula el gradiente (regularizado)
def gradiente(Theta, X, Y, _lambda):
    H = sigmoide(np.matmul(X, Theta))
    m = np.shape(X)[0]

    return (1/m) * np.matmul(np.transpose(X), H - Y) + ((_lambda / m ) * Theta)


def LR_Optimize(Theta, X, Y, reg):
    result=opt.fmin_tnc(func=coste, x0 = Theta, fprime = gradiente, args=(X, Y, reg))
    return result[0]

def LR_HyperparameterTuning(Theta, X, Y, Xval, Yval, reg):
    scores = np.zeros(reg.size)

    for i in range(reg.size):
        thetaOpt = LR_Optimize(Theta, X, Y, reg[i])
        aux = np.matmul(Xval, thetaOpt)
        auxSig = sigmoide(aux)
        scores[i] = accuracy_score(Yval, auxSig)
        print("Reg:{0}: {1}".format(reg[i], scores[i]))
    
    print(scores)
    best = np.max(scores)
    print("maxAcc: ", best)