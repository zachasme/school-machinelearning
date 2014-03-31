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

X = dataset.X.values[1:100,:]
y = dataset.y[1:100]
N,M = X.shape
classNames = dataset.classNames[1:100]
attributeNames = dataset.attributeNames




# exercise 7.1.2

from pylab import *
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation


# Maximum number of neighbors
L=30

CV = cross_validation.LeaveOneOut(N)
errors = np.zeros((N,L))
i=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l, warn_on_equidistant=False);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1
    
# Plot the classification error rate
figure()
plot(100*sum(errors,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()
savefig("classy-as-knn.png")