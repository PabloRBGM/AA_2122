import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.index_tricks import AxisConcatenator
import sklearn.svm as s_svm
from pandas.io.parsers import read_csv

# ajustamos los datos de entrada para nuestro modelo
def clean_data(data):
    # utilizamos el header para hacer un diccionario con cada 
    dataHeader = data.columns.values
    # indice de la columna y luego lo quitamos
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
                 header['track_name'],
                 header['obtained_date']], 1)


    #rehacemos el diccionario sin las columnas que hemos quitado
    dataHeader = np.delete(dataHeader,
                [header['instance_id'],
                 header['track_name'],
                 header['obtained_date']], 0)
    # indice de la columna y luego lo quitamos
    header = dict(enumerate(dataHeader.flatten(),0))
    header = dict((value,key) for key, value in header.items())

    # pasamos los strings a numeros
    data_ok[:, header['artist_name']] = np.unique(data_ok[:, header['artist_name']], return_inverse=True)[1]
    data_ok[:, header['key']] = np.unique(data_ok[:, header['key']], return_inverse=True)[1]
    data_ok[:, header['mode']] = np.unique(data_ok[:, header['key']], return_inverse=True)[1]
    data_ok[:, header['music_genre']] = np.unique(data_ok[:, header['music_genre']], return_inverse=True)[1]

    return data_ok, header

# devuelve los indices del genero
def index_songs_of_genre(Y, genre):
    return np.where(Y == genre)[0]

def main():
    data = read_csv("music_genre.csv")
  
    data_ok,headerDict = clean_data(data)

    # comprobamos cuantos generos hay y cuantas canciones de cada genero
    genres = np.unique(data_ok[:, headerDict['music_genre']])

    # creamos las conjuntos de casos vacios
    Xtrain = np.empty((0, data_ok.shape[1]))
    Ytrain = np.empty((0,1))
    Xval = np.empty((0, data_ok.shape[1]))
    Yval = np.empty((0,1))
    Xtest = np.empty((0, data_ok.shape[1]))
    Ytest = np.empty((0,1))

    # construimos los conjuntos de entrenamiento, validacion y test
    for i in range(len(genres)):
        songs = index_songs_of_genre(data_ok, genres[i])

        Xtrain = np.append(Xtrain, data_ok[songs[:300]], axis=0)
        Ytrain = np.append(Ytrain, np.full(300, genres[i]))
        Xval =  np.append(Xval, data_ok[songs[300:400]], axis=0)
        Yval =  np.append(Yval, np.full(100, genres[i]))
        Xtest =  np.append(Xtest, data_ok[songs[400:450]], axis=0)
        Ytest =  np.append(Ytest, np.full(50, genres[i]))
        #print("{0}: {1}".format(genres[i],np.size(songs_of_genre(Y, genres[i]))))

    


main()