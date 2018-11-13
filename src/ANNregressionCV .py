# exercise 8.2.6
##
# Author Janus Bastian Lansner S145349
# 
from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from scipy import stats

# exercise 6.2.1
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

from standardizer import standardize

#run file reader we will get X and lables
from file_reader import *
import numpy as np
from scipy.io import loadmat
from sklearn import tree
import graphviz
import math

def hidden_n_selector(X,y,cvf=10):
    ''' Function performs feature selection for linear regression model using
        'cvf'-fold cross validation. The process starts with empty set of
        features, and in every recurrent step one feature is added to the set
        (the feature that minimized loss function in cross-validation.)

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds

        Returns:
        selected_features   indices of optimal set of features
        features_record     boolean matrix where columns correspond to features
                            selected in subsequent steps
        loss_record         vector with cv errors in subsequent steps
        
        Example:
        selected_features, features_record, loss_record = ...
            feature_selector_lr(X_train, y_train, cvf=10)
            
    ''' 
    y = y.squeeze() #ÆNDRING JLH #9/3
    best_error = None
    #best_train_error = None
    best_n = -1
    for hidden_n in range(1,X.shape[1]):
        temp_error = ann_validate(X,y,hidden_n,cvf)#Use test error, not train
        if best_error is None:
            best_error = temp_error
            best_n = hidden_n
        if temp_error<best_error :
            best_error = temp_error
            best_n = hidden_n
    
    return best_n, best_error
    # first iteration error corresponds to no-feature estimator
    #if loss_record is None:
    #    loss_record = np.array([np.square(y-y.mean()).sum()/y.shape[0]])
    #if features_record is None:
    #    features_record = np.zeros((X.shape[1],1))

    # Add one feature at a time to find the most significant one.
    # Include only features not added before.
    #selected_features = features_record[:,-1].nonzero()[0]
    #min_loss = loss_record[-1]
    #if display is 'verbose':
    #    print(min_loss)
    #best_feature = False
    #for hidden_n in range(0,X.shape[1]):
    #    if np.where(selected_features==feature)[0].size==0:
    #        trial_selected = np.concatenate((selected_features,np.array([feature])),0).astype(int)
            # validate selected features with linear regression and cross-validation:
    #        trial_loss = ann(X[:,trial_selected],y,cvf)
    #        if trial_loss<min_loss:
    #            min_loss = trial_loss 
    #            best_feature = feature

    # If adding extra feature decreased the loss function, update records
    # and go to the next recursive step
    #if best_feature is not False:
    #    features_record = np.concatenate((features_record, np.array([features_record[:,-1]]).T), 1)
    #    features_record[best_feature,-1]=1
    #    loss_record = np.concatenate((loss_record,np.array([min_loss])),0)
    #    selected_features, features_record, loss_record = hidden_n_selector(X,y,cvf,features_record,loss_record)
        
    # Return current records and terminate procedure
    #return selected_features, features_record, loss_record
        

def ann_validate(X,y,n_hidden_units,cvf=3):
    
    # Parameters for neural network classifier
    #n_hidden_units = 2      # number of hidden units
    n_train = 2             # number of networks trained in each k-fold
    learning_goal = 5     # stop criterion 1 (train mse to be reached)
    max_epochs = 64         # stop criterion 2 (max epochs in training)
    show_error_freq = 30     # frequency of training status updates
    
    # K-fold crossvalidation
    K = cvf                  # only three folds to speed up this example
    CV = model_selection.KFold(K,shuffle=True)
    
    # Variable for classification error
    errors = np.zeros(K)*np.nan
    error_hist = np.zeros((max_epochs,K))*np.nan
    bestnet = list()
    #k=0
    ''' Validate linear regression model using 'cvf'-fold cross validation.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns MSE averaged over 'cvf' folds.

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds        
    '''
    y = y.squeeze()
    CV = model_selection.KFold(n_splits=cvf, shuffle=True)
    validation_error=np.empty(cvf)
    f=0
    for train_index, test_index in CV.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # extract training and test set for current CV fold
    
        best_train_error = np.inf
        for i in range(n_train):
            print('Training network {0}/{1}...'.format(i+1,n_train))
            # Create randomly initialized network with 2 layers
            ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
            if i==0:
                bestnet.append(ann)
                # train network
            train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            if train_error[-1]<best_train_error:
                bestnet[f]=ann
                best_train_error = train_error[-1]
            error_hist[range(len(train_error)),f] = train_error

        print('Best train error: {0}...'.format(best_train_error))
        y_est = bestnet[f].sim(X_test).squeeze()
        errors[f] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
        #k+=1
        #m = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        validation_error[f] = np.square(y_est-y_test).sum()/y_test.shape[0]
        f=f+1
    return errors.mean()

#def hidden_n_selector(X,y,cvf=10);
#def ann_validate(X,y,n_hidden_units,cvf=10)

# get standardized data
#standardizedData = standardize(data)
mat_data = standardize(data)

# Load Matlab data file and extract variables of interest
#mat_data = loadmat('../Data/wine.mat')
#X = mat_data['X']
#y = mat_data['y'].astype(int).squeeze()
#C = mat_data['C'][0,0]
#M = mat_data['M'][0,0]
#N = mat_data['N'][0,0]

X = mat_data[:,range(1,7)]
#print(X)
y = mat_data[:,0].squeeze()
#for obj in range(0,len(y)) :
#    if(y[obj] != 0) :
#        y[obj] = 1
#print(y)
attributeNames = [
    'age',
    'sex M=0,F=1',
    'trestbps',
    'chol',
    'fbs',
    'thalach',
    'exang'
    ]
classNames = [
        'Not sick',
        'Sick'
        ]
N, M = X.shape

                
## Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
#Y = stats.zscore(X,0);
#U,S,V = np.linalg.svd(Y,full_matrices=False)
#V = V.T
##Components to be included as features
#k_pca = 3
#X = X @ V[:,0:k_pca]
#N, M = X.shape

K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)
selected_n = [0]*K
errors = np.zeros(K)*np.nan
errors_test = np.zeros(K)*np.nan
Error_train_nofeatures = np.empty((K,1))#Baseline
Error_test_nofeatures = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
k=0
bestnet = list()
for train_index, test_index in CV.split(X):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 3
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.power(y_train-y_train.mean(),2).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.power(y_test-y_test.mean(),2).sum()/y_test.shape[0]
    #np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    selected_n[k], errors[k] = hidden_n_selector(X_train,y_train,internal_cross_validation)
    
    #
    # Parameters for neural network classifier
    #n_hidden_units = 2      # number of hidden units
    n_train = 2             # number of networks trained in each k-fold
    learning_goal = 5     # stop criterion 1 (train mse to be reached)
    max_epochs = 64         # stop criterion 2 (max epochs in training)
    show_error_freq = 5     # frequency of training status updates
    
    # K-fold crossvalidation
    K = 10                  # only three folds to speed up this example
    CV = model_selection.KFold(K,shuffle=True)
    
    # Variable for classification error
    #errors2 = np.zeros(K)*np.nan
    error_hist = np.zeros((max_epochs,K))*np.nan
    #k=0
    ''' Validate linear regression model using 'cvf'-fold cross validation.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns MSE averaged over 'cvf' folds.

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds        
    '''
    y = y.squeeze()
    #CV = model_selection.KFold(n_splits=10, shuffle=True)
    validation_error=np.empty(10)
    #f=0       
    # extract training and test set for current CV fold
   
    best_train_error = np.inf
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i+1,n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]]*M, [selected_n[k], 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            bestnet.append(ann)
            # train network
        train_error_local = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error_local[-1]<best_train_error:
            bestnet[k]=ann
            best_train_error = train_error_local[-1]
        error_hist[range(len(train_error_local)),k] = train_error_local
    print('Best train error: {0}...'.format(best_train_error))
    #Error_test_fs[k]
    someval = bestnet[k].sim(X_test).squeeze()
    Error_test_fs[k] = np.power(someval-y_test,2).sum().astype(float)/y_test.shape[0]
    #
    k+=1
    
    #best_train_error = np.inf
    #for i in range(n_train):
        #print('Training network {0}/{1}...'.format(i+1,n_train))
        # Create randomly initialized network with 2 layers
        #ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        #if i==0:
        #    bestnet.append(ann)
        # train network
        #train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        #if train_error[-1]<best_train_error:
        #    bestnet[k]=ann
        #    best_train_error = train_error[-1]
        #    error_hist[range(len(train_error)),k] = train_error

    #print('Best train error: {0}...'.format(best_train_error))
    #y_est = bestnet[k].sim(X_test).squeeze()
    #errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    #k+=1
    #break

# Print the average least squares error
#print('Mean-square error: {0}'.format(np.mean(errors)))
print("n")
print(selected_n)
print("errors")
print(errors)
print("Results")
# Display results
print('\n')
#print('Linear regression without feature selection:\n')
#print('- Training error: {0}'.format(Error_train.mean()))
#print('- Test error:     {0}'.format(Error_test.mean()))
#print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
#print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
#print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(errors.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-errors.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))
#figure(figsize=(6,7));
#subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');
#subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
#figure(figsize=(6,7));
#subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
#subplot(2,1,2); plot((y_est-y_test)); title('Last CV-fold: prediction error (est_y-test_y)'); 
#show()

print('Ran Exercise 8.2.6')

#% The weights if the network can be extracted via
#bestnet[0].layers[0].np['w'] # Get the weights of the first layer
#bestnet[0].layers[0].np['b'] # Get the bias of the first layer


