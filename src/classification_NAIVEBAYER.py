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

X = dataset.X.values
y = dataset.y
N = dataset.N
M = dataset.M
classNames = dataset.classNames
attributeNames = dataset.attributeNames




# exercise 7.2.4

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
k=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=est_prior)
    cls = nb_classifier
    nb_classifier.fit(X_train, y_train)
    y_est_prob = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob,1)
    
    errors[k] = np.sum(y_est.ravel()!=y_test.ravel(),dtype=float)/y_test.shape[0]
    k+=1
    
# Plot the classification error rate
print('Error rate: {0}%'.format(100*mean(errors)))


figure()
plot(100*errors/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()
savefig("classy-as-knn.png")

print(nb_classifier)