#%%
import numpy as np
import codecs
import sklearn.svm as s_svm

from get_vocab_dict import getVocabDict
from process_email import email2TokenList
from sklearn.metrics import accuracy_score

def hyperparameter_tuning(X, y, Xval, yval):
    C = 0.01
    n_iter = 8
    scores = np.zeros((n_iter, n_iter))
    for i in range(n_iter):
        sigma = 0.01
        for j in range(n_iter):
            svm = s_svm.SVC(C=C, kernel='rbf', gamma= (1 / (2*sigma**2)))
            svm.fit(X, y)
            scores[i][j] = accuracy_score(yval, svm.predict(Xval))
            #print("accuracy: ",  accuracy_score(yval, svm.predict(Xval)))
            sigma *= 3
            print("C:{0}, sigma:{1} : {2}".format(C,sigma, scores[i][j]))
        C *= 3
    print(scores)
    print("maxAcc: ", np.max(scores))


def build_mat(X, y, result, dir, ini, end, vocab_dict):
    n = len(vocab_dict)
    for i in range(ini, end):
        case = np.zeros(n)
        email_contents = codecs.open('{0}/{1:04d}.txt'.format(dir,i), 
                                    'r', encoding='utf-8', errors='ignore').read()
        email = email2TokenList(email_contents)
        for j in range(len(email)):
            if vocab_dict.get(email[j], False):
                case[vocab_dict[email[j]] - 1] = 1
        X = np.vstack((X, case))
        y = np.hstack((y, result))
    
    return X, y

def main():
    vocab_dict = getVocabDict()
    n = len(vocab_dict)

    # Construimos la matriz de entrenamiento
    X = np.empty((0, n))
    y = np.empty(0)
    
    X, y = build_mat(X, y, 1, "spam", 1, 250, vocab_dict)
    X, y = build_mat(X, y, 0, "easy_ham", 1, 1259, vocab_dict)
    X, y = build_mat(X, y, 0, "hard_ham", 1, 125, vocab_dict)

    # Construimos la matriz de validacion
    Xval = np.empty((0, n))
    yval = np.empty(0)
    
    Xval, yval = build_mat(Xval, yval, 1, "spam", 250, 350, vocab_dict)
    Xval, yval = build_mat(Xval, yval, 0, "easy_ham", 1259, 2250, vocab_dict)
    Xval, yval = build_mat(Xval, yval, 0, "hard_ham", 125, 200, vocab_dict)
        
    # Construimos la matriz de test
    Xtest = np.empty((0, n))
    ytest = np.empty(0)

    Xtest, ytest = build_mat(Xtest, ytest, 1, "spam", 350, 400, vocab_dict)
    Xtest, ytest = build_mat(Xtest, ytest, 0, "easy_ham", 2250, 2551, vocab_dict)
    Xtest, ytest = build_mat(Xtest, ytest, 0, "hard_ham", 200, 250, vocab_dict)
    
    hyperparameter_tuning(X, np.ravel(y), Xval, np.ravel(yval))
    #hyperparameter_tuning(X, np.ravel(y), Xtest, np.ravel(ytest))



main()
# %%