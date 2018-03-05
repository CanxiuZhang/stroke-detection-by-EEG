import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from tempfile import TemporaryFile
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import scikitplot as skplt

#dataset = np.load('FeatureData1.npy')  # feature dataset (None,19)
#testdata = np.load('TUHFeature.npy')
train = np.load('TrainFeature1.npy') # feature dataset (None,19)
test = np.load('TestFeature1.npy') # feature dataset (None,19)
#print(dataset.shape)

#data = dataset
#tdata = testdata
### 10-fold cross validation

# X = data[:, 0:22]
# X = preprocessing.normalize(X, norm='l2', axis=1)
#
# y = data[:, 22]
# for i in range(y.shape[0]):
#     if y[i] == 0:
#         y[i] = -1

X_train = train[:, 0:22]
X_train = preprocessing.normalize(X_train, norm='l2', axis=0, copy=True, return_norm=False)
y_train = train[:, 22]
for i in range(y_train.shape[0]):
    if y_train[i] == 0:
        y_train[i] = -1

X_test = test[:, 0:22]
X_test = preprocessing.normalize(X_test, norm='l2', axis=0, copy=True, return_norm=False)
y_test = test[:, 22]
for i in range(y_test.shape[0]):
    if y_test[i] == 0:
        y_test[i] = -1

# TX = tdata[:, 0:23]
# Ty = tdata[:, 23]
# for i in range(Ty.shape[0]):
#     if Ty[i] == 0:
#         Ty[i] = -1

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_cv = []
test_cv = []

#kf = KFold(n_splits=10, shuffle=True)
# kf.get_n_splits(X)
# cverror = []
# for train_index, test_index in kf.split(X):
#     print("TEST:", test_index, "Test number:", len(test_index))
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(75,), random_state=15)
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)
y_result = clf.predict(X_test)
print(y_score.shape)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
train_cv.append(train_score)
test_cv.append(test_score)

print("mean accuracy (train): ", np.mean(train_cv))
#print("stand error (train):", np.std(train_cv))
print("mean accuracy (test): ", np.mean(test_cv))
#print("stand error (test):", np.std(test_cv))
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(25,), random_state=45)
    # clf = RandomForestClassifier(n_estimators=200)
    # clf.fit(X_train, y_train)
    # y_pred = []
    # for i in range(X_test.shape[0]):
    #     y_pred.append(clf.predict(X_test[i].reshape(1, -1)))
    #
    # y_pred = np.array(y_pred)
    # roc = roc_auc_score(y_test, y_pred)
    #
    # print("mean ROc: ", roc)
    # count = 0
    #
    # for i in range(X_test.shape[0]):
    #     if (y_pred[i] == y_test[i]):
    #         count = count + 1
    #
    # accuracy = count / (X_test.shape[0])
    # cverror.append(accuracy)
#
# print("mean accuracy: ", np.mean(cverror))
# skplt.metrics.plot_roc_curve(y_test, y_score)
#
#
# skplt.metrics.plot_precision_recall_curve(y_test, y_score)
#
# skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_result)
#
# plt.show()

