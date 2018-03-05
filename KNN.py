from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#data = np.load('featureData.npy') # feature dataset (None,19)
#data = np.load('RawfeatureData.npy') # feature dataset (None,19)
data_train = np.load('TrainFeature.npy')
data_test = np.load('TestFeature.npy')


#neigh = KNeighborsClassifier(n_neighbors=3)
# X = data[:, 0:18]
# y = data[:, 18]
X_train = data_train[:, 0:18]
y_train = data_train[:, 18]

X_test = data_test[:, 0:18]
y_test = data_test[:, 18]
cvtest = []
cvtrain = []
auc = []

clf = KNeighborsClassifier(n_neighbors=4) ### KNN classifier (with hyperparameters)

clf.fit(X_train, y_train)
accuracy = clf.score(X_train, y_train)
cvtrain.append(accuracy)
accuracy = clf.score(X_test, y_test)
cvtest.append(accuracy)
#print(clf.summary())
y_pred = []
for i in range(X_test.shape[0]):
    y_pred.append(clf.predict(X_test[i].reshape(1, -1)))
    # ytrain.append(clf.predict(X_train[i].reshape(1,-1)))
y_pred = np.array(y_pred)
auc.append(roc_auc_score(y_pred, y_test))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('TN', tn, 'FP', fp, 'FN', fn, 'TP', tp)

print("Train accuracy is", np.mean(cvtrain))
print("Test accuracy is", np.mean(cvtest))
print("auc", np.mean(auc))