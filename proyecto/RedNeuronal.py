import numpy as np
import scipy.optimize as opt
import math
from sklearn.metrics import accuracy_score
import sklearn.neural_network as s_nn

def sigmoide(z):
    return 1/(1+ math.e**(-z))

def coste(theta1,theta2, X, Y ):
    a_1,a_2,H = forward_propagate(X, theta1, theta2)
    m = np.shape(X)[0]
    coste = (- 1 / (m)) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H + 1e-9))
    
    return coste

def coste_reg(theta1, theta2, X, Y, _lambda):
    
    a_1, a_2, H = forward_propagate(X, theta1, theta2)
    theta1 = theta1[:,1:]
    theta2 = theta2[:,1:]
    m = np.shape(X)[0]
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
    a_2 = sigmoide(z_2)
    a_2 = np.hstack([np.ones([m, 1]), a_2]) 
    z_3 = np.matmul(a_2, np.transpose(theta2)) 
    h=sigmoide(z_3)
    return a_1, a_2, h

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):
    # Roll de las matrices de teta
    Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas +1)], (num_ocultas, (num_entradas+1)))
    Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas +1):], (num_etiquetas, (num_ocultas+1)))
    
    # Calculo del gradiente
    A1, A2, H=forward_propagate(X,Theta1, Theta2)
    m=np.shape(X)[0]

    Delta1 = np.zeros(np.shape(Theta1))
    Delta2 = np.zeros(np.shape(Theta2))
    for t in range(m):
        a1t = A1[t,:] 
        a2t = A2[t,:]
        ht = H[t,:] 
        yt = Y[t]

        d3t = ht - yt 
        d2t = np.dot(np.transpose(Theta2),d3t)*(a2t*(1-a2t))

        Delta1 = Delta1 + np.dot(d2t[1:,np.newaxis],a1t[np.newaxis,:])
        Delta2 = Delta2 + np.dot(d3t[:,np.newaxis],a2t[np.newaxis,:])
        
    
    Theta1ceros = Theta1[:,1:]
    Theta1ceros = np.hstack([np.zeros([np.shape(Theta1)[0], 1]), Theta1ceros]) 
    D1 = (1/m) * (Delta1 + reg * Theta1ceros)
    
    Theta2ceros = Theta2[:,1:]
    Theta2ceros = np.hstack([np.zeros([np.shape(Theta2)[0], 1]), Theta2ceros]) 
    D2 = (1/m) * (Delta2 + reg * Theta2ceros)

    # Unroll del gradiente
    gradientVec = np.concatenate((np.ravel(D1),np.ravel(D2)))
    
    jVal = coste_reg(Theta1, Theta2, X, Y, reg)
    return (jVal, gradientVec)

def training(params, num_entradas, num_ocultas, num_etiquetas, X, Y, reg, nIter):
    
    res = opt.minimize(
        fun=backprop, 
        x0=params, 
        args=( num_entradas, num_ocultas, num_etiquetas, X, Y, reg), 
        method='TNC',
        jac=True,
        options={'maxiter': nIter})
   
    
    return res

# Calcula el porcentaje de éxito de nuestro modelo multi-clase
# params son los pesos de la red neuronal entrenados
def evaluacion(X,y, params, num_entradas, num_ocultas, num_etiquetas):
    Theta1 = np.reshape(params[:num_ocultas * (num_entradas +1)], (num_ocultas, (num_entradas+1)))
    Theta2 = np.reshape(params[num_ocultas * (num_entradas +1):], (num_etiquetas, (num_ocultas+1)))

    A1, A2, H=forward_propagate(X,Theta1, Theta2)
    # lr_acc = accuracy_score(y, H.astype(int))
    MaxIndex = np.zeros(np.shape(H)[0])
    for i in range(np.shape(H)[0]):
        MaxIndex[i]=np.argmax(H[i])
        
    lr_acc = np.sum(np.ravel(y.astype(int)) == MaxIndex.astype(int))
    return lr_acc


def NN_HyperparameterTuning(num_ocultas, X, Y, Xval, Yval, reg, numIter):
    scores = np.zeros(reg.size)
    
    num_entradas = np.shape(X)[1]
    num_etiquetas = np.shape(Y)[1]
    
    _low = -0.12
    _high = -_low
    
    Theta1 = np.random.uniform(low=_low, high=_high, size=(num_ocultas, num_entradas + 1) )
    #Theta1 = np.zeros((num_ocultas, num_entradas + 1))
    Theta2 = np.random.uniform(low=_low, high=_high, size=(num_etiquetas, num_ocultas + 1) )
    #Theta2 = np.zeros((num_etiquetas, num_ocultas + 1))

    params = np.concatenate((np.ravel(Theta1), np.ravel(Theta2)))
      
    for i in range(reg.size):
        res = training(params, num_entradas, num_ocultas, num_etiquetas, X, Y, reg[i], numIter)
        scores[i] = evaluacion(Xval, Yval, res.x, num_entradas, num_ocultas, num_etiquetas)
        print("Accuracy of Neural Network with C={0} : {1}".format(reg[i], scores[i]))
    
    print(scores)
    best = np.max(scores)
    print("maxAcc: ", best)

def NN_Evaluate(num_entradas, num_ocultas, num_etiquetas, X, Y, Xtest, Ytest, reg, numIter):
    _low = -0.12
    _high = -_low
    Theta1 = np.random.uniform(low=_low, high=_high, size=(num_ocultas, num_entradas + 1) )
    #Theta1 = np.zeros((num_ocultas, num_entradas + 1))
    Theta2 = np.random.uniform(low=_low, high=_high, size=(num_etiquetas, num_ocultas + 1) )
    #Theta2 = np.zeros((num_etiquetas, num_ocultas + 1))
    
    params = np.concatenate((np.ravel(Theta1), np.ravel(Theta2)))
    res = training(params, num_entradas, num_ocultas, num_etiquetas, X, Y, reg, numIter)
    return evaluacion(Xtest, Ytest, res.x, num_entradas, num_ocultas, num_etiquetas)

def SKLearn_NN_HyperparameterTuning(num_ocultas, X, Y, Xval, Yval, reg, numIter):
    scores = np.zeros(reg.size)

    for i in range(len(reg)):
        nn = s_nn.MLPClassifier(hidden_layer_sizes=num_ocultas, max_iter=numIter, solver='sgd', alpha=reg[i])
        model = nn.fit(X, Y)
        ypred = nn.predict(Xval)
        scores[i] = accuracy_score(Yval, ypred)
        print("Accuracy of Neural Network with C={0} : {1}".format(reg[i], scores[i]))
    print(scores)
    best = np.max(scores)
    return best, reg[np.where(scores == best)]

def SKLearn_NN_Evaluate(num_ocultas, X, Y, Xtest, Ytest, reg, numIter):
    nn = s_nn.MLPClassifier(hidden_layer_sizes=num_ocultas, max_iter=numIter, solver='lbfgs', alpha=reg)
    model = nn.fit(X, Y)
    ypred = nn.predict(Xtest)
    
    return accuracy_score(Ytest, ypred)