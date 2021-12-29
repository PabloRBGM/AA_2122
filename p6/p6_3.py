#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sklearn.preprocessing as sp
import sklearn.svm as s_svm
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

def visualize_boundary(X, y, svm, file_name):
    x1 = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
    x2 = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.close()

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
        C *= 3
    print(scores)
    print("maxAcc: ", np.max(scores))


def main():
    data1 = loadmat('ex6data3.mat')
    X  = data1['X']
    Xval = data1['Xval']
    y = data1['y']
    yval = data1['yval']

    hyperparameter_tuning(X, np.ravel(y), Xval, np.ravel(yval))


main()
# %%

