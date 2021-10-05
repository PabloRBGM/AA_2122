import time
import random as rnd
import numpy as np
#import scipy.integrate
import matplotlib.pyplot as plt

def cuadrado(x):
    return x * x
    
#funcion lenta usando bucles
def integra_mc(fun, a, b, num_puntos=1000):
    print("Integral iterativa")
    tic = time.process_time()
    nDebajo = 0
    aux = np.linspace(a,b, 500)
    funct = fun(aux)
    m = max(funct)

    for i in range(0, num_puntos):
        point = [rnd.randint(a, b), rnd.randint(0, m)]
        if point[1] < fun(point[0]):
            nDebajo+=1

    result = (nDebajo / num_puntos) * (b - a) * m
    
    toc = time.process_time()

    print(result)
    #print(1000 * (toc - tic))

    return 1000 * (toc - tic)

#funcion rapida usando vectores
def integra_mc_np(fun, a, b, num_puntos=1000):
    print("Integral vectorial")
    tic = time.process_time()
    nDebajo = 0
    aux = np.linspace(a,b, 500)
    funct = fun(aux)
    m = max(funct)

    pointsX = np.random.randint(a,b + 1, num_puntos)
    pointsY = np.random.randint(0,m + 1, num_puntos)
    nDebajo = sum(pointsY < (fun(pointsX)))
    result = (nDebajo / num_puntos) * (b - a) * m

    toc = time.process_time()
    
    print(result)
    #print(1000 * (toc - tic))
    return 1000 * (toc - tic)

def compara_tiempos():
    num_puntos_grafica = 20
    sizes=np.linspace(100,10000000000,num_puntos_grafica)
    times = []
    times_np = []
    for size in range(1, num_puntos_grafica +1):        
        print(2**size )#numero de puntos en esa iteracion
        times += [integra_mc(cuadrado, 0, 2,2**size)]
        times_np += [integra_mc_np(cuadrado, 0, 2,2**size)]

    plt.figure()
    plt.scatter(sizes, times, c='red', label='bucle')
    plt.scatter(sizes, times_np, c='blue' , label='vector')
    plt.legend()
    plt.savefig('time.png')

#print(integra_mc(cuadrado, 0, 2,1000000))
#print(integra_mc_np(cuadrado, 0, 2,1000000))
#print(scipy.integrate.quad(cuadrado, 0,2)[0])
compara_tiempos()