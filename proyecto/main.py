import numpy as np
from pandas.io.parsers import read_csv
from SVM import SVM_HyperparameterTuning, SVM_Evaluate
from RegresionLogistica import LR_HyperparameterTuning, LR_Evaluate
from RedNeuronal import NN_HyperparameterTuning, SKLearn_NN_HyperparameterTuning, NN_Evaluate, SKLearn_NN_Evaluate
from sklearn.preprocessing import StandardScaler

def normalize_data(data):
    scaler = StandardScaler()
    normalizedData = scaler.fit_transform(data)
    return normalizedData
# ajustamos los datos de entrada para nuestro modelo
def clean_data(data):
    # utilizamos el header para hacer un diccionario con cada 
    dataHeader = data.columns.values

    header = dict(enumerate(dataHeader.flatten(),0))
    header = dict((value,key) for key, value in header.items())

    # quitamos las filas nulas
    data = data.dropna()

    # quitamos duplicados si los hay
    data.duplicated().any()
    duplicatedCol = data[data.duplicated()].T.columns
    data.drop(duplicatedCol, inplace=True)    

    # convertimos los datos a matriz de numpy
    data_ok = data.to_numpy()

    # la columna 1 'artist_name' tiene valores vacios y los eliminamos
    data_ok = np.delete(data_ok, np.where(data_ok[:,header['artist_name']] == "empty_field")[0], 0)
    # la columna 14 'tempo' tiene valores erroneos y no podemos
    # hacer una aproximacion del valor que podria tener, eliminamos esos casos
    data_ok = np.delete(data_ok, np.where(data_ok[:,header['tempo']] == "?")[0], 0)
    # esta columna tiene valores string que tenemos que convertir a float
    data_ok[:,header['tempo']] = data_ok[:,header['tempo']].astype("float") # TODO: luego convertimos todo a float
    # la columan de 'duration_ms' tiene valores de -1, quitamos esos casos
    # data_ok = np.delete(data_ok, np.where(data_ok[:,header['duration_ms']] == -1.0)[0], 0)

    # quitar columnas no necesarias (id, nombre de la pista, fecha)
    data_ok = np.delete(data_ok,
                [header['instance_id'],
                 header['artist_name'],
                 header['track_name'],
                 header['obtained_date'],
                 #header['popularity'],
                 #header['acousticness'],
                 #header['danceability'],
                 header['duration_ms'],
                 #header['instrumentalness'],
                 #header['key'],
                 #header['loudness'],
                 #header['energy'],
                 #header['liveness'],
                 #header['mode'],
                 #header['speechiness'],
                 header['tempo'],
                 #header['valence']
                 ], 1)


    # rehacemos el diccionario sin las columnas que hemos quitado
    dataHeader = np.delete(dataHeader,
                [header['instance_id'],
                 header['artist_name'],
                 header['track_name'],
                 header['obtained_date'],
                 #header['popularity'],
                 #header['acousticness'],
                 #header['danceability'],
                 header['duration_ms'],
                 #header['instrumentalness'],
                 #header['key'],
                 #header['loudness'],
                 #header['energy'],
                 #header['liveness'],
                 #header['mode'],
                 #header['speechiness'],
                 header['tempo'],
                 #header['valence']]
                 ], 0)

    header = dict(enumerate(dataHeader.flatten(),0))
    header = dict((value,key) for key, value in header.items())

    # pasamos los strings a numeros
    #data_ok[:, header['artist_name']] = np.unique(data_ok[:, header['artist_name']], return_inverse=True)[1]
    data_ok[:, header['key']] = np.unique(data_ok[:, header['key']], return_inverse=True)[1]
    data_ok[:, header['mode']] = np.unique(data_ok[:, header['mode']], return_inverse=True)[1]
    data_ok[:, header['music_genre']] = np.unique(data_ok[:, header['music_genre']], return_inverse=True)[1]

    return data_ok, header

# devuelve los indices del genero
def index_songs_of_genre(Y, genre):
    return np.where(Y == genre)[0]


# Calcula el porcentaje de éxito de nuestro modelo multi-clase
def evaluacion(y, classification):
    MaxIndex=np.zeros(np.shape(classification)[0])

    for i in range(np.shape(classification)[0]):
        MaxIndex[i]=np.argmax(classification[i]) 
        
    Num= np.sum(np.ravel(y) == MaxIndex)
    Porcentaje=Num/np.shape(y)[0] * 100
    print("Succes: {0}%".format(Porcentaje))

def main():
    data = read_csv("music_genre.csv")
  
    data_ok,headerDict = clean_data(data)
    data_ok = data_ok.astype(float)
    data_normalized = normalize_data(data_ok)
    # comprobamos cuantos generos hay y cuantas canciones de cada genero
    genres = np.unique(data_ok[:, headerDict['music_genre']])

    # creamos las conjuntos de casos vacios
    Xtrain = np.empty((0, data_ok.shape[1] - 1))
    Ytrain = np.empty(0)
    Xval = np.empty((0, data_ok.shape[1] - 1 ))
    Yval = np.empty(0)
    Xtest = np.empty((0, data_ok.shape[1] - 1))
    Ytest = np.empty(0)


    #numTrain = 300
    #numVal = 100
    #numTest = 50
    numTrain = 2940
    numVal = 840
    numTest = 420

    # construimos los conjuntos de entrenamiento, validacion y test
    for i in range(len(genres)):
        songs = index_songs_of_genre(data_ok[:,-1], genres[i])
        ini = 0
        end = numTrain

        aux = data_ok[songs[ini:end]]
        aux2 = aux[:,-1]
        Xtrain = np.append(Xtrain, aux[:,:-1], axis=0)
        Ytrain = np.append(Ytrain, aux2, axis=0)
        ini = end
        end += numVal

        aux = data_ok[songs[ini:end]]
        Xval =  np.append(Xval, aux[:,:-1], axis=0)
        Yval =  np.append(Yval, aux[:, -1], axis=0)
        ini = end
        end += numTest

        aux = data_ok[songs[ini:end]]
        Xtest =  np.append(Xtest, aux[:,:-1], axis=0)
        Ytest =  np.append(Ytest, aux[:, -1], axis=0)

    print("Training X cases: {0}".format(Xtrain.shape))
    print("Training Y cases: {0}".format(Ytrain.shape))
    print("Validation X cases: {0}".format(Xval.shape))
    print("Validation Y cases: {0}".format(Yval.shape))
    print("Test X cases: {0}".format(Xtest.shape))
    print("Test Y cases: {0}".format(Ytest.shape))    

    # Clasificacion con Regresion logistica
    LR_Study(len(genres), Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)
    # Clasificacion con SVC
    SVC_Study(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)
    # Clasificacion con Red Neuronal
    NN_Study(len(genres), 100, 200, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, True)


def LR_Study( num_etiquetas, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
    reg = np.array([0, 0.1, 0.5, 1, 5, 10, 25, 100, 250])
    # Elegimos el valor de regularizacion mas apto
    acc, bestReg = LR_HyperparameterTuning(num_etiquetas, Xtrain, Ytrain, Xval, Yval, reg)
    print("MaxAcc: {0}, Reg: {1}".format(acc, bestReg))
    # Probamos el modelo con los casos de prueba para medir el exito
    print("Exito Regresion Logistica: {0}".format(LR_Evaluate(num_etiquetas, Xtrain, Ytrain, Xtest, Ytest, bestReg)))

def SVC_Study(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
    Cs = np.array([0.01, 0.03, 0.09, 0.27, 0.81, 2.43, 7.29, 21.87, 65.61])
    sigmas = np.array([0.01, 0.03, 0.09, 0.27, 0.81, 2.43, 7.29, 21.87, 65.61])
    # Elegimos el valor de regularizacion mas apto
    acc, bestC, bestSigma =  SVM_HyperparameterTuning(Xtrain, Ytrain, Xval, Yval, Cs, sigmas)
    print("MaxAcc: {0}, C: {1}, Sigma: {2}".format(acc, bestC, bestSigma))
    # Probamos el modelo con los casos de prueba para medir el exito
    print("Exito SVC: {0}".format(SVM_Evaluate(Xtrain, Ytrain, Xtest, Ytest, bestC, bestSigma)))

def onehot(Y, n_classes):
    onehot = np.zeros((np.shape(Y)[0], n_classes))
    for i in range(np.shape(Y)[0]):
        onehot[i][Y[i].astype(int)] = 1
    return onehot

def NN_Study(n_classes, num_ocultas, num_iter, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, use_sklearn):
    reg = np.array([0.1, 0.5, 1, 5, 10, 25, 100, 250])

    ytrain_onehot = onehot(Ytrain, n_classes)
    yval_onehot = onehot(Yval, n_classes)
    ytest_onehot = onehot(Ytest, n_classes)

    if(use_sklearn):
        acc, bestReg = SKLearn_NN_HyperparameterTuning(num_ocultas, Xtrain, ytrain_onehot, Xval, yval_onehot, reg, num_iter)
        print("MaxAcc: {0}, Reg: {1}".format(acc, bestReg))
        print("Exito Red Neuronal: {0}".format(NN_Evaluate(num_ocultas, Xtrain, ytrain_onehot, Xtest, ytest_onehot, reg, num_iter)))
    else:
        acc, bestReg = NN_HyperparameterTuning(num_ocultas, Xtrain, ytrain_onehot, Xval, yval_onehot, reg, num_iter)
        print("MaxAcc: {0}, Reg: {1}".format(acc, bestReg))
        print("Exito Red Neuronal: {0}".format(SKLearn_NN_Evaluate(num_ocultas, Xtrain, ytrain_onehot, Xtest, ytest_onehot, reg, num_iter)))

main()