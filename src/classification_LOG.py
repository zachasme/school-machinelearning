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
#	'autoTheft', 'autoTheftPerPop',
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




# exercise 7.2.4

from pylab import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

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

# K-fold crossvalidation
K = 10
CV = cross_validation.KFold(N,K,shuffle=True)
cls = None
errors = np.zeros(K)
Error_logreg = np.zeros(K)
k=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
        
    model3 = lm.logistic.LogisticRegression(C=N)
    model3 = model3.fit(X_train, y_train)
    y_logreg = np.mat(model3.predict(X_test)).T
    Error_logreg[k] = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)

    k+=1
    
# Plot the classification error rate
print('Error rate: {0}%'.format(100*mean(Error_logreg)))


figure()
plot(100*Error_logreg/N)
xlabel('LogisticRegression')
ylabel('Classification error rate (%)')
show()
savefig("classy-as-log.png")
