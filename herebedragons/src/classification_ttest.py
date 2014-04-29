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
            if cell == '?':
                row.append(np.nan)
            else:
                row.append(cell)
    data.append(row)

names = next(namereader)
drop_columns = [
    'State',
    'countyCode', 'communityCode', 'communityname',
    'fold',
    'murders', 'murdPerPop',
    'rapes', 'rapesPerPop',
    'robberies', 'robbbPerPop',
    'assaults', 'assaultPerPop',
    'burglaries', 'burglPerPop',
    'larcenies', 'larcPerPop',
#   'autoTheft', 'autoTheftPerPop',
    'arsons', 'arsonsPerPop',
    'violentPerPop',
    'nonViolPerPop',
]






from DataSet import *
from PCA import *

import matplotlib.pyplot as plt

dataset = DataSet(data, names, class_column='State', drop_columns=drop_columns, fix_missing=FixMissing.FILLMEAN, rescale=Rescale.NORMALIZE)

y = np.mat(np.zeros((len(names),1)))

TO = 50
X = dataset.X.values[0:TO,:]
y = dataset.y[0:TO]
N, M = X.shape
classNames = dataset.classNames
attributeNames = dataset.attributeNames[0:TO]




# exercise 6.2.1

from pylab import *
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation, tree
from scipy import stats
from pylab import *
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from pylab import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation


# Naive Bayes classifier parameters
alpha = 1.0         # additive parameter (e.g. Laplace correction)
est_prior = True   # uniform prior (change to True to estimate prior from data)


# Maximum number of neighbors
L=30


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = cross_validation.KFold(N,K,shuffle=True)
#CV = cross_validation.StratifiedKFold(y.A.ravel(),k=K)

# Initialize variables
Error_nb = np.empty((K,1))
Error_kn = np.empty((K,1))
Error_logreg = np.empty((K,1))
n_tested=0

k=0
for train_index, test_index in CV:
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit and evaluate K Nearest Neighbour
    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        model = KNeighborsClassifier(n_neighbors=l, warn_on_equidistant=False);
        model.fit(X_train, y_train);
        y_kn = np.mat(model.predict(X_test)).T
        Error_kn[k] = 100*(y_kn!=y_test).sum().astype(float)/len(y_test)
    
    # Fit and evaluate Naive Bayes
    model2 = MultinomialNB(alpha=alpha, fit_prior=est_prior)
    model2.fit(X_train, y_train)
    y_nb = np.mat(model2.predict(X_test)).T
    Error_nb[k] = 100*(y_nb!=y_test).sum().astype(float)/len(y_test)

    #y_est_prob = nb_classifier.predict_proba(X_test)
    #y_est = np.argmax(y_est_prob,1)
    #errors[k] = np.sum(y_est.ravel()!=y_test.ravel(),dtype=float)/y_test.shape[0]

    # Fit and evaluate Logistic Regression classifier
    model3 = lm.logistic.LogisticRegression(C=N)
    model3 = model3.fit(X_train, y_train)
    y_logreg = np.mat(model3.predict(X_test)).T
    Error_logreg[k] = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)

    k+=1

# Use T-test to check if classifiers are significantly different
[tstatistic, pvalue] = stats.ttest_ind(Error_kn,Error_nb)
if pvalue<=0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.bmat('Error_kn, Error_nb'))
xlabel('KNN vs Naive Bayes')
ylabel('Cross-validation error [%]')




# Use T-test to check if classifiers are significantly different
[tstatistic, pvalue] = stats.ttest_ind(Error_kn,Error_logreg)
if pvalue<=0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.bmat('Error_kn, Error_logreg'))
xlabel('KNN vs LogisticRegression')
ylabel('Cross-validation error [%]')




# Use T-test to check if classifiers are significantly different
[tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_nb)
if pvalue<=0.05:
    print('Classifiers are significantly different. (p={0})'.format(pvalue[0]))
else:
    print('Classifiers are not significantly different (p={0})'.format(pvalue[0]))        
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.bmat('Error_logreg, Error_nb'))
xlabel('LogisticRegression vs Naive Bayes')
ylabel('Cross-validation error [%]')

show()