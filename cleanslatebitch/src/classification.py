import pylab as pl


from Framework.DataSet import *

crime = DataSet(datafile='../data/raw.csv', nominals=['state','communityname','countyCode','communityCode'])

crime = crime.drop_columns([
	'state', 'communityname',
	#'countyCode', 'communityCode',
	#'fold',
	'murders', 'murdPerPop',
	'rapes', 'rapesPerPop',
	'robberies', 'robbbPerPop',
	'assaults', 'assaultPerPop',
	'burglaries', 'burglPerPop',
	'larcenies', 'larcPerPop',
	'autoTheft', 'autoTheftPerPop',
	'arsons', 'arsonsPerPop',
	'ViolentCrimesPerPop',
	'nonViolPerPop',
])

crime = crime.normalize()
crime = crime.fix_missing(fill_mean=True)


crime = crime.classIn('countyCode')

print(type(crime.X))
print(crime.y)
X=crime.X
y=crime.y


# exercise 7.1.2

from pylab import *
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation


# Maximum number of neighbors
N=10
L=30
kn = None
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
        kn = knclassifier
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

print(kn)