import numpy as np
# warning is not logged here. Perfect for clean unit test output
# with np.errstate(divide='ignore'):
#     np.float64(1.0) / 0.0
import matplotlib.pyplot as plt
from numpy.lib.index_tricks import AxisConcatenator
import sklearn.svm as s_svm
from pandas.io.parsers import read_csv
from SVM import SVM_HyperparameterTuning
from RegresionLogistica import LR_HyperparameterTuning
from RedNeuronal import NN_HyperparameterTuning
#----
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.svm as s_svm
from sklearn.metrics import accuracy_score
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
    data_ok[:,header['tempo']] = data_ok[:,header['tempo']].astype("float")

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
    # comprobamos cuantos generos hay y cuantas canciones de cada genero
    genres = np.unique(data_ok[:, headerDict['music_genre']])

    # creamos las conjuntos de casos vacios
    Xtrain = np.empty((0, data_ok.shape[1] - 1))
    Ytrain = np.empty(0)
    Xval = np.empty((0, data_ok.shape[1] - 1 ))
    Yval = np.empty(0)
    Xtest = np.empty((0, data_ok.shape[1] - 1))
    Ytest = np.empty(0)


    numTrain = 300
    numVal = 100
    numTest = 50
    #numTrain = 2940
    #numVal = 840
    #numTest = 420

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

    ytrain_onehot = np.zeros((np.shape(Ytrain)[0], len(genres)))
    for i in range(np.shape(Ytrain)[0]):
        ytrain_onehot[i][Ytrain[i].astype(int)] = 1

    yval_onehot = np.zeros((np.shape(Yval)[0], len(genres)))
    for i in range(np.shape(Yval)[0]):
        yval_onehot[i][Yval[i].astype(int)] = 1

    # Decidimos hyperparametros
    reg = np.array([0, 1, 10, 25])
    #flout = Xtrain.astype(float)
    NN_HyperparameterTuning(25, Xtrain, ytrain_onehot, Xval, yval_onehot, reg, 70)
    # LR_HyperparameterTuning(Xtrain, Ytrain, Xval, Yval, reg)
    # SVM_HyperparameterTuning(Xtrain, Ytrain, Xval, Yval)
    # Test

    # Validacion

    # Test
    #model6 = 'Support Vector Classifier'
    #svc = s_svm.SVC(kernel = 'linear', C = 2)
    #svc.fit(Xtrain, Ytrain)
    #ypred = svc.predict(Xtest)
    #svc_cm = confusion_matrix(Ytest, ypred)
    #svc_acc = accuracy_score(Ytest, ypred)
    #print('Confusion Matrix')
    #print(svc_cm)
    #print('\n')
    #print(f'Accuracy of {model6} : {svc_acc * 100} \n')
    #print(classification_report(Ytest, ypred))

main()