import numpy as np
import codecs
import sklearn.svm as s_svm
from sklearn.metrics import accuracy_score

# ajustamos los datos de entrada para nuestro modelo
def SVM_HyperparameterTuning(X, y, Xval, yval, Cs, sigmas):
    n = len(Cs)
    m = len(sigmas)
    scores = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            svm = s_svm.SVC(C=Cs[i], kernel='linear', gamma= (1 / (2* sigmas[j]**2)))
            svm.fit(X, y)
            scores[i][j] = accuracy_score(yval, svm.predict(Xval))
            print("C:{0}, sigma:{1} : {2}".format(Cs[i], sigmas[j], scores[i][j]))
    print(scores)
    best = np.max(scores)
    aux = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    return best, Cs[aux[0]], sigmas[aux[1]]

def SVM_Evaluate(Xtrain, Ytrain, Xtest, Ytest, C, sigma):
    svm = s_svm.SVC(C=C, kernel='linear', gamma= (1 / (2* sigma**2)))
    svm.fit(Xtrain, Ytrain)
    return accuracy_score(Ytest, svm.predict(Xtest))