import numpy as np
import scipy.optimize as opt
import math

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
    Num= np.sum(np.ravel(Y) == MaxIndex)
    Porcentaje=Num/nCases
    return Porcentaje

def oneVsAll(X, y, num_etiquetas, reg):
    """
    oneVsAll entrena varios clasificadores por regresion logistica con termino de regularizacion
    'reg' y devuelve el resultado en una matriz, donde la fila i-esima correspone
    al clasificador de la etiqueta i-esima
    """
    theta_mat = np.zeros((num_etiquetas, np.shape(X)[1] + 1))
    m = np.shape(X)[0]
    X1s = np.hstack([np.ones([m, 1]), X])
    Theta = np.zeros(np.shape(X1s)[1])
    for i in range(num_etiquetas):
        y_i = (y==i)
        y_i = np.ravel(y_i)
        print(np.shape(y_i))
        result = opt.fmin_tnc(func=coste, x0 = Theta, fprime = gradiente, args=(X1s, y_i, reg))
        theta_mat[i] = result[0]
    
    return theta_mat    

def LR_HyperparameterTuning(num_etiquetas, X, Y, Xval, Yval, reg ):
    scores = np.zeros(reg.size)
    m = np.shape(Xval)[0]
    Xval1s = np.hstack([np.ones([m, 1]), Xval])
    for i in range(reg.size):
        scores[i] = evaluacion(Xval1s, Yval, oneVsAll(X, Y, num_etiquetas, reg[i]))
        print("Reg:{0}: {1}".format(reg[i], scores[i]))
    
    print(scores)
    best = np.max(scores)
    return best, reg[np.where(scores == best)]

def LR_Evaluate(num_etiquetas, X, Y, Xtest, Ytest, reg):
    m = np.shape(Xtest)[0]
    Xtest1s = np.hstack([np.ones([m, 1]), Xtest])
    return evaluacion(Xtest1s, Ytest, oneVsAll(X, Y, num_etiquetas, reg))
