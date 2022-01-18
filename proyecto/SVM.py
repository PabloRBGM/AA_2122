import numpy as np
import codecs
import sklearn.svm as s_svm
from sklearn.metrics import accuracy_score

# ajustamos los datos de entrada para nuestro modelo
def SVM_HyperparameterTuning(X, y, Xval, yval):
    C = 0.01
    n_iter = 8
    scores = np.zeros((n_iter, n_iter))
    for i in range(n_iter):
        sigma = 0.01
        for j in range(n_iter):
            svm = s_svm.SVC(C=C, kernel='rbf', gamma= (1 / (2*sigma**2)))
            svm.fit(X, y)
            scores[i][j] = accuracy_score(yval, svm.predict(Xval))
            sigma *= 3
            print("C:{0}, sigma:{1} : {2}".format(C,sigma, scores[i][j]))
        C *= 3
    print(scores)
    print("maxAcc: ", np.max(scores))

def SVM_Classify():
    print("not implemented")