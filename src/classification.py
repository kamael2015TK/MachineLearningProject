from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from standardizer import *
import numpy as np
from scipy.io import loadmat
from classification_methods import getDecisionTree, resetStateHolder
from sklearn import model_selection
import math


#run file reader we will get X and lables
from file_reader import *
standardizedData = standardize(data)

# feature to prodict 
targetFeatureIndex = 7

# creating range to remove the feature we want to prodict 
if targetFeatureIndex == len(standardizedData[0]) : 
    featureRange = range(0, len(standardizedData[0]) - 1)
else :
    featureRange = list(range(0,targetFeatureIndex)) + list(range(targetFeatureIndex + 1, len(standardizedData[0]) ))

# spliting data
X = standardizedData[:, featureRange]
y = data[:, targetFeatureIndex].squeeze()
N, M = X.shape
y = arrayToBinary(y)
del attributeNames[len(attributeNames)-1]
classNames = [
        'Not sick',
        'Sick'
        ]

N, M = X.shape

outer_loop = 10
inner_loop = 10

CV_outer = model_selection.KFold(n_splits=outer_loop,shuffle=True)

des_tree_gen_errors = np.empty(outer_loop)

k = 0
for train_index_o, test_index_o in CV_outer.split(X):
    
    # extract training and test set for current CV fold
    X_train_outer = X[train_index_o,:]
    y_train_outer = y[train_index_o]
    X_test_outer = X[test_index_o,:]
    y_test_outer = y[test_index_o]

    # splitting training data for inner loop
    CV_inner = model_selection.KFold(n_splits=inner_loop,shuffle=True)

    # init controll variables
    startDepth = 5
    dessisionTreeDepth = range(startDepth, startDepth+inner_loop)
    i = 0
    des_tree_error_rate = 100
    des_tree_best_model = None 
    des_tree_best_depth = None
    for train_index_i, test_index_i in CV_inner.split(X_train_outer) :
        X_train_inner = X[train_index_i,:]
        y_train_inner = y[train_index_i]
        X_test_inner = X[test_index_i,:]
        y_test_inner = y[test_index_i]
        confusionMatrix, model = getDecisionTree(
            data_x=X_train_inner,
            data_y=y_train_inner,
            test_x=X_test_inner,
            test_y=y_test_inner,
            attributeNames=attributeNames,
            split=2,
            depth=dessisionTreeDepth[i])
        accuracy = 100*confusionMatrix.diagonal().sum()/confusionMatrix.sum()
        error_rate = 100-accuracy
        if des_tree_error_rate > error_rate:
            best_error_rate = error_rate
            des_tree_best_model = model
            des_tree_best_depth = i
        i += 1
    des_tree_gen_errors
    y_prediction = des_tree_best_model.predict(X_test_outer)
    des_tree_gen_conf_matrix = confusion_matrix(y_test_outer, y_prediction)
    accuracy = 100*des_tree_gen_conf_matrix.diagonal().sum()/des_tree_gen_conf_matrix.sum()
    error_rate = 100-accuracy
    des_tree_gen_errors[k] = error_rate
    k += 1

des_tree_gen_error =  des_tree_gen_errors.sum()/len(des_tree_gen_errors)
print(des_tree_gen_error)





X_train = X[range(0,math.floor(len(X)/2))]
X_test = X[range(math.floor(len(X)/2),len(X))]

y_train = y[range(0,math.floor(len(y)/2))]
y_test = y[range(math.floor(len(y)/2),len(y))]
C = len(classNames)

figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,2], styles[c])


Knearest=5
dist=2

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=Knearest, p=dist)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,0], X_test[class_mask,2], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,2], 'kx', markersize=8)
title('Synthetic data classification - KNN')

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est)
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy
figure(2)
imshow(cm, cmap='binary', interpolation='None')
colorbar()
xticks(range(C)); yticks(range(C))
xlabel('Predicted class'); ylabel('Actual class')
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate))

show()
