#%%
import numpy as np
import scipy.optimize as opt
import math
from scipy.io import loadmat
#ir caso a caso de X viendo que predice cada theta y quedarnos con el mayor y luego comprobar si es realmente ese
#por los 5000 casos, probar si es 0,1,2...9, ver cual es mayor y lo guardamos, luego comprobar si es cierto

# Calcula la funcion sigmoide 
def sigmoide(z):
    return 1/(1+math.e**(-z))

#  Calcula el coste sin regularizar
def coste(theta1,theta2, X, Y ):
    a_1,a_2,H = forward_propagate(X, theta1, theta2)
    m = np.shape(X)[0]
    # usamos la formula con las traspuestas
    coste = (- 1 / (m)) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H + 1e-9))
    
    return coste

def coste_reg(theta1, theta2, X, Y, _lambda):
    
    a_1, a_2, H = forward_propagate(X, theta1, theta2)
    theta1 = theta1[:,1:]
    theta2 = theta2[:,1:]
    m = np.shape(X)[0]
    # usamos la formula con las traspuestas
    coste = (- 1 / (m)) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H + 1e-9))
    
    regularizacion = (_lambda /(2*m)) * (np.sum(np.power(theta1,2)) + np.sum(np.power(theta2,2)))
    coste += regularizacion
    
    return coste

    # def coste_reg(theta1, theta2, X, Y, _lambda):
    # a_1, a_2, H = forward_propagate(X, theta1, theta2)
    # theta1 = theta1[:,1:]
    # theta2 = theta2[:,1:]
    # m = np.shape(X)[0]
    # # usamos la formula con las traspuestas
    # coste = (- 1 / (m)) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H + 1e-9))
    
    # regularizacion = (_lambda /(2*m)) * (np.sum(np.power(theta1,2)) + np.sum(np.power(theta2,2)))
    # coste += regularizacion
    
    # return coste

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
    a_2=sigmoide(z_2)
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
        a1t = A1[t,:] #(401,)
        a2t = A2[t,:] #(26,)
        ht = H[t,:] #(10,)
        yt = Y[t] #(10,)

        d3t = ht - yt #(10,)
        d2t = np.dot(np.transpose(Theta2),d3t)*(a2t*(1-a2t)) #26

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

def trainint(params, num_entradas, num_ocultas, num_etiquetas, X, Y, reg, nIter):
    
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
    MaxIndex=np.zeros(np.shape(H)[0])
    for i in range(np.shape(H)[0]):
        MaxIndex[i]=np.argmax(H[i]) + 1#mas uno por como esta ordenado,el 0 equivale a 1... el 9 a un 10
        
    Num= np.sum(np.ravel(y) == MaxIndex)
    Porcentaje=Num/np.shape(y)[0] * 100
    print(Porcentaje)

def main():
    data = loadmat('ex4data1.mat')
    Y = np.ravel(data[ 'y' ])
    X = data [ 'X' ]
    
    _low = -0.12
    _high = -_low

    Theta1 = np.random.uniform(low=_low, high=_high, size=(25,401) )
    Theta2 = np.random.uniform(low=_low, high=_high, size=(10,26) )
   
    num_entradas = np.shape(X)[1]
    num_ocultas = np.shape(Theta1)[0]
    num_etiquetas = np.shape(Theta2)[0]


    m = len(Y)
    #Para tener mas ordenada la matriz y. Para evitar pasarnos con los indices
    y = (Y-1)
    y_onehot = np.zeros((m,num_etiquetas)) # 5000 x 10
    for i in range(m):
        y_onehot[i][y[i]] = 1
    params = np.concatenate((np.ravel(Theta1), np.ravel(Theta2)))
    res = trainint(params,num_entradas,num_ocultas, num_etiquetas, X,y_onehot, 1, 70 )

    evaluacion( X, Y, res.x,num_entradas,num_ocultas, num_etiquetas)

main()
# %%


# diez clasificadores, aplicar la regresion logistica 10 veces,
# en p2, las etiquetas son 0,1; la y del fichero va de 1 a 10
# tenemos que cambiar la y con n valores por una que solo tenga 0s y 1s
