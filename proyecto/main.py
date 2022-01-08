import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as s_svm
from pandas.io.parsers import read_csv

def songs_of_genre(Y, genre):
    return np.where(Y == genre)

def main():
    data = read_csv("music_genre.csv")


    # quitamos las filas nulas
    data = data.dropna()

    # quitamos duplicados si los hay
    # print(data.shape)
    data.duplicated().any()
    duplicatedCol = data[data.duplicated()].T.columns
    data.drop(duplicatedCol, inplace=True)
    # print(data.shape)
    
    # quitar columnas no necesarias (id, fecha, nombre de la pista)

    # convertimos los datos a matriz de numpy
    data = data.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]

    # comprobamos cuantos generos hay y cuantas canciones de cada genero
    # construimos los conjuntos de entrenamiento, validacion y test
    genres = np.unique(Y)
    for i in range(len(genres)):
        print(np.size(songs_of_genre(Y, genres[i])))
    

main()