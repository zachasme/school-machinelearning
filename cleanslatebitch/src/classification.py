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
	#'ViolentCrimesPerPop',
	'nonViolPerPop',
])

crime = crime.normalize()
crime = crime.fix_missing(fill_mean=True)
crime = crime.discretize('ViolentCrimesPerPop',2)

crime = crime.classIn('ViolentCrimesPerPop')

print(type(crime.X))
print(crime.y)
X=crime.X

dataset=crime
y = crime.y

X = dataset.X
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
K = 100
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

print(nb_classifier)