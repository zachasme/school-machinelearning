import csv
import numpy as np

DATADIR = '../data/'

"""read attribute data and names from comma-separated files"""
namefile = open(DATADIR + 'communities.names')
datafile = open(DATADIR + 'communities.data')
#datafile = open('data/communities_norm.data')

datareader = csv.reader(datafile, delimiter=',', quotechar='|')
namereader = csv.reader(namefile, delimiter=',', quotechar='|')

data = []
for rowreader in datareader:
	row = []
	for cell in rowreader:
		try:
			row.append(float(cell))
		except:
			row.append(np.nan)
	data.append(row)

names = next(namereader)
drop_columns = [
	'State', 'countyCode', 'communityCode', 'communityname',
	'murders', 'murdPerPop',
	'rapes', 'rapesPerPop',
	'robberies', 'robbbPerPop',
	'assaults', 'assaultPerPop',
	'burglaries', 'burglPerPop',
	'larcenies', 'larcPerPop',
#	'autoTheft', 'autoTheftPerPop',
	'arsons', 'arsonsPerPop',
	'violentPerPop',
	'nonViolPerPop',
]


from DataSet import *
from PCA import *

oli = DataSet(data, names, drop_columns=drop_columns, fix_missing=FixMissing.DROPOBJECTS, rescale=Rescale.NORMALIZE)

#print("\n\nstd:",   oli.X.std())
#print("\n\nmean:",  oli.X.mean())
#print("\n\nrange:", oli.X.max()-oli.X.min())

#Oli.PCA(oli)

#regression
from pylab import *
from sklearn import cross_validation
from toolbox_02450 import feature_selector_lr, bmplot
import sklearn.linear_model as lm

X = oli.X
attributeNames = X.columns.values

import pandas as pd
import neurolab as nl
import scipy.linalg as linalg

# Split dataset into features and target vector
y = X.autoTheftPerPop.values

X = X.loc[:, :'policBudgetPerPop'].values

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict autotheftperpop
y_est = model.predict(X)
residual = y_est - y

# Display scatter plot
figure()
subplot(2, 1, 1)
plot(y, y_est, '.')
xlabel('autoTheftPerPop (true)'); ylabel('autoTheftPerPop (estimated)');
subplot(2, 1, 2)
hist(residual, 40)

#show()

N = oli.N()
M = oli.M()

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = cross_validation.KFold(N,K,shuffle=True)

# Parameters for neural network classifier
n_hidden_units = 2      # number of hidden units
n_train = 2             # number of networks trained in each k-fold
learning_goal = 100     # stop criterion 1 (train mse to be reached)
max_epochs = 64         # stop criterion 2 (max epochs in training)
show_error_freq = 5     # frequency of training status updates
# Variable for classification error
errors = np.zeros(K)
error_hist = np.zeros((max_epochs,K))
bestnet = list()

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV:
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation)
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    m = lm.LinearRegression().fit(X_train[:,selected_features], y_train)
    Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
    Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]

    best_train_error = 1e100
    for i in range(n_train):
        X_train = X[train_index,:]
        y_train = y[train_index,:]
        X_test = X[test_index,:]
        y_test = y[test_index,:]

        print('Training network {0}/{1}...'.format(i+1,n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[0, 1]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            bestnet.append(ann)
        # train network
        train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error[-1]<best_train_error:
            bestnet[k]=ann
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)),k] = train_error

    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].sim(X_test)
    errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]

    figure(k)
    subplot(1,2,1)
    plot(range(1,len(loss_record)), loss_record[1:])
    xlabel('Iteration')
    ylabel('Squared error (crossvalidation)')    
    
    subplot(1,3,3)
    bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
    clim(-1.5,0)
    xlabel('Iteration')

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


figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')


# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual
f=2 # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]
m = lm.LinearRegression().fit(X[:,ff], y)

y_est= m.predict(X[:,ff])
residual=y-y_est

figure(k+1)
title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
for i in range(0,len(ff)):
   subplot(2,ceil(len(ff)/2.0),i+1)
   plot(X[:,ff[i]],residual,'.')
   xlabel(attributeNames[ff[i]])
   ylabel('residual error')

print('Mean-square error: {0}'.format(mean(errors)))
figure(k+2);
subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');
subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
figure(k+3);
subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
subplot(2,1,2); plot(y_est-y_test); title('Last CV-fold: prediction error (est_y-test_y)');

show()  