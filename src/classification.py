from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
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
knn_gen_errors_array = np.empty(outer_loop)
logreg_gen_errors_array = np.empty(outer_loop)
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
    knn_neighbors = [ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 ]
    i = 0
    des_tree_error_rate = 100
    des_tree_best_model = None 
    des_tree_best_depth = 0
    knn_t_error_rate = 100
    knn_best_model = 0
    knn_neighbors_count = 0
    logreg_error_rate = 100
    logreg_best_model = 0

    for train_index_i, test_index_i in CV_inner.split(X_train_outer) :
        X_train_inner = X[train_index_i,:]
        y_train_inner = y[train_index_i]
        X_test_inner = X[test_index_i,:]
        y_test_inner = y[test_index_i]
        # des tree
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
            des_tree_best_depth = dessisionTreeDepth[i]
        # KNN 
        knnclassifier = KNeighborsClassifier(n_neighbors=knn_neighbors[i], p=2)
        knnclassifier.fit(X_train_inner, y_train_inner)
        y_est = knnclassifier.predict(X_test_inner)
        knn_cm = confusion_matrix(y_test_inner, y_est)
        knn_accuracy = 100*knn_cm.diagonal().sum()/knn_cm.sum(); 
        knn_error_rate = 100-knn_accuracy
        if knn_t_error_rate > knn_error_rate : 
            knn_best_model = knnclassifier 
            knn_neighbors_count = knn_neighbors[i]

        # logistic regression
        logreg = LogisticRegression()
        logreg.fit(X_train_inner, y_train_inner)
        logreg_y_pred = logreg.predict(X_test_inner)
        logreg_cm = confusion_matrix(y_test_inner, logreg_y_pred)
        logreg_accuracy = 100*logreg_cm.diagonal().sum()/logreg_cm.sum(); 
        logreg_t_error_rate = 100-logreg_accuracy

        if logreg_error_rate > logreg_t_error_rate : 
            logreg_best_model = logreg

                

    ## gen error calc des tree 
    y_prediction = des_tree_best_model.predict(X_test_outer)
    des_tree_gen_conf_matrix = confusion_matrix(y_test_outer, y_prediction)
    accuracy = 100*des_tree_gen_conf_matrix.diagonal().sum()/des_tree_gen_conf_matrix.sum()
    error_rate = 100-accuracy
    des_tree_gen_errors[k] = error_rate

    ## knn ger error 
    knn_y_prodict = knn_best_model.predict(X_test_outer)
    gen_error_con_matrix = confusion_matrix(y_test_outer, knn_y_prodict)
    knn_gen_accuracy = 100*gen_error_con_matrix.diagonal().sum()/gen_error_con_matrix.sum()
    knn_gen_error_rate = 100-knn_gen_accuracy
    knn_gen_errors_array[k] = knn_gen_error_rate

    ## log reg error
    #  
    logreg_y_prodict = logreg_best_model.predict(X_test_outer)
    logreg_gen_error_con_matrix = confusion_matrix(y_test_outer, logreg_y_prodict)
    logreg_gen_accuracy = 100*logreg_gen_error_con_matrix.diagonal().sum()/logreg_gen_error_con_matrix.sum()
    logreg_gen_error_rate = 100-logreg_gen_accuracy
    logreg_gen_errors_array[k] = logreg_gen_error_rate
    k += 1

des_tree_gen_error_average =  des_tree_gen_errors.sum()/len(des_tree_gen_errors)
print(des_tree_gen_error_average)
knn_gen_errors_array_average = knn_gen_errors_array.sum() / len(knn_gen_errors_array)
print(knn_gen_errors_array_average)
logreg_gen_error_rate_average = logreg_gen_errors_array.sum() / len(logreg_gen_errors_array)
print(logreg_gen_error_rate_average)





# Plot the classfication results
#styles = ['ob', 'or', 'og', 'oy']
#for c in range(C):
#    class_mask = (y_est==c)
#    plot(X_test[class_mask,0], X_test[class_mask,2], styles[c], markersize=10)
#    plot(X_test[class_mask,0], X_test[class_mask,2], 'kx', markersize=8)
#title('Synthetic data classification - KNN')

# Compute and plot confusion matrix

#figure(2)
#imshow(cm, cmap='binary', interpolation='None')
#colorbar()
#xticks(range(C)); yticks(range(C))
#xlabel('Predicted class'); ylabel('Actual class')
#title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate))

#show()
