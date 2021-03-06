#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as s_svm
from scipy.io import loadmat

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

def main():
    data1 = loadmat('ex6data1.mat')
    X  = data1['X']
    # X1val = data1['Xval']
    # X1test = data1['Xtest']
    y = data1['y']
    # y1val = data1['yval']
    # y1test = data1['ytest']

    svm = s_svm.SVC(C=100.0, kernel='linear')
    svm.fit(X, y)
    visualize_boundary(X, np.ravel(y), svm, "6_1B.png")

main()
# %%

