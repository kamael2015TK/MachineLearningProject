from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from standardizer import *

#run file reader we will get X and lables
from file_reader import *
import numpy as np
from scipy.io import loadmat
from sklearn import tree
import graphviz
import math

mat_data = standardize(data)

X = mat_data[:,range(0,7)]
y = data[:,7].squeeze()

y = arrayToBinary(y)

del attributeNames[len(attributeNames)-1]
print(attributeNames)
classNames = [
        'Not sick',
        'Sick'
        ]

N, M = X.shape


# exercise 5.1.2

# Fit regression tree classifier, Gini split criterion, no pruning
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=10)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames)
#graphviz.render('dot','png','tree_gini',quiet=False)
src=graphviz.Source.from_file('tree_gini.gvz')


# Load Matlab data file and extract variables of interest
#mat_data = loadmat('../Data/synth1.mat')
#X = mat_data['X']
X_train = X[range(0,math.floor(len(X)/2))]
X_test = X[range(math.floor(len(X)/2),len(X))]
#y = mat_data['y'].squeeze()
y_train = y[range(0,math.floor(len(y)/2))]
y_test = y[range(math.floor(len(y)/2),len(y))]
#attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
#classNames = [name[0][0] for name in mat_data['classNames']]
#N, M = X.shape
C = len(classNames)


# Plot the training data points (color-coded) and test data points.
figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,2], styles[c])


# K-nearest neighbors
Knearest=5

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=Knearest, p=dist);
knclassifier.fit(X_train, y_train);
y_est = knclassifier.predict(X_test);


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,0], X_test[class_mask,2], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,2], 'kx', markersize=8)
title('Synthetic data classification - KNN');

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

show()
