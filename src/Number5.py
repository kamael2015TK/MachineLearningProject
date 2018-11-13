##
# Author Janus Bastian Lansner S145349

from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from scipy import stats
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

mat_data = standardize(data)
X = mat_data[:,range(1,7)]
y = mat_data[:,0].squeeze()

N, M = X.shape


#n_hidden_units = 2      # number of hidden units
n_train = 2             # number of networks trained in each k-fold
learning_goal = 100     # stop criterion 1 (train mse to be reached)
max_epochs = 64         # stop criterion 2 (max epochs in training)
show_error_freq = 5     # frequency of training status updates


# K-fold crossvalidation
K = 5                   # only three folds to speed up this example
CV = model_selection.KFold(K,shuffle=True)

# Variable for classification error
k=0
ANNerror = np.zeros(K)
ANNhiddenn = np.zeros(K)
errors_outer = np.zeros(K)*np.nan
error_hist_outer = np.zeros((max_epochs,K))*np.nan
bestBestNet = list()
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

for train_index, test_index in CV.split(X):
    print("OUTERFOLD")
    K2 = 3
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    #Baseline
    #Train...
    #Test...
    #ANN  
    resultn = -1
    bestError = None
    #Train...
    for hidden_n in range(1,X.shape[1]):
        errors = np.zeros(K2)*np.nan
        error_hist = np.zeros((max_epochs,K2))*np.nan
        bestnet = list()
        n_train = 2             # number of networks trained in each k-fold
        learning_goal = 5     # stop criterion 1 (train mse to be reached)
        max_epochs = 64         # stop criterion 2 (max epochs in training)
        show_error_freq = 30     # frequency of training status updates
        
        # K-fold crossvalidation
        CV = model_selection.KFold(K2,shuffle=True)
        
        # Variable for classification error
        errors = np.zeros(K2)*np.nan
        error_hist = np.zeros((max_epochs,K2))*np.nan
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
        CV = model_selection.KFold(n_splits=K2, shuffle=True)
        validation_error=np.empty(K2)
        f=0
        for train_index_inner, test_index_inner in CV.split(X_train):
            X_train_inner = X[train_index_inner]
            y_train_inner = y[train_index_inner]
            X_test_inner = X[test_index_inner]
            y_test_inner = y[test_index_inner]
            
            # extract training and test set for current CV fold
        
            best_train_error = np.inf
            for i in range(n_train):
                print('Training network {0}/{1}...'.format(i+1,n_train))
                # Create randomly initialized network with 2 layers
                ann = nl.net.newff([[-3, 3]]*M, [hidden_n, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
                if i==0:
                    bestnet.append(ann)
                    # train network
                train_error = ann.train(X_train_inner, y_train_inner.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
                if train_error[-1]<best_train_error:
                    bestnet[f]=ann
                    best_train_error = train_error[-1]
                error_hist[range(len(train_error)),f] = train_error
    
            print('Best train error: {0}...'.format(best_train_error))
            y_est = bestnet[f].sim(X_test_inner).squeeze()
            errors[f] = np.power(y_est-y_test_inner,2).sum().astype(float)/y_test.shape[0]
            #k+=1
            #m = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
            validation_error[f] = np.square(y_est-y_test_inner).sum()/y_test_inner.shape[0]
            f=f+1
        
        
        tempError = np.mean(errors)
        if bestError is None:
            bestError = np.mean(errors)
            resultn = 1
        if tempError < bestError :
            bestError = np.mean(errors)
            resultn = 1
    #Test...
    best_train_error = np.inf
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i+1,n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]]*M, [resultn, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            bestBestNet.append(ann)
            # train network
            train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error[-1]<best_train_error:
            bestBestNet[k]=ann
            best_train_error = train_error[-1]
            error_hist_outer[range(len(train_error)),k] = train_error
    
    #print('Best train error: {0}...'.format(best_train_error))
    y_est = bestBestNet[k].sim(X_test).squeeze()
    #errors[f] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    #k+=1
    #m = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    ANNerror[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    ANNhiddenn[k] = resultn
    
    
    
    #FeatureSelection
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
      #  figure(k)
      #  subplot(1,2,1)
      #  plot(range(1,len(loss_record)), loss_record[1:])
      #  xlabel('Iteration')
       # ylabel('Squared error (crossvalidation)')    
        
       # subplot(1,3,3)
        #bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        #clim(-1.5,0)
        #xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))
    k+=1

# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))


print(ANNerror)
print(ANNhiddenn)

z = (ANNerror-Error_test_fs)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers (ANN and FEATURE) are not significantly different')        
else:
    print('Classifiers (ANN and FEATURE) are significantly different.')
    
z = (ANNerror-Error_test)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers (ANN and BASELINE) are not significantly different')        
else:
    print('Classifiers (ANN and BASELINE) are significantly different.')
    
    
    
z = (Error_test_fs-Error_test)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers (FEATURE and BASELINE) are not significantly different')        
else:
    print('Classifiers (FEATURE and BASELINE) are significantly different.')