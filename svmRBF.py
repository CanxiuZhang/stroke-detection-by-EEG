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
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

train = np.load('TrainFeature1.npy') # feature dataset (None,19)
test = np.load('TestFeature1.npy') # feature dataset (None,19)

#data = np.load('featureData.npy')

# X = data[:, 0:22]
# X = preprocessing.normalize(X, norm='l2', axis=0, copy=True, return_norm=False)
# y = data[:, 22]
# for i in range(y.shape[0]):
#     if y[i] == 0:
#         y[i] = -1
# print(X.shape)
# lb = preprocessing.LabelBinarizer()
# #lb.fit([1, -1])
# y = lb.fit_transform(y)
# print(y.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# kf = KFold(n_splits=30)
# kf.get_n_splits(X)

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
train_cv = []
test_cv = []
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#clf = svm.SVC(C=100, gamma=100, kernel='linear', probability=True)
clf = KNeighborsClassifier(n_neighbors=4)  ### KNN classifier (with hyperparameters)
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

#fpr, tpr, thresholds = roc_curve(y_test, y_score)
#print(thresholds.shape)
#roc_auc = auc(y_test, y_score)

# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

skplt.metrics.plot_roc_curve(y_test, y_score)


skplt.metrics.plot_precision_recall_curve(y_test, y_score)

skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_result)

plt.show()

