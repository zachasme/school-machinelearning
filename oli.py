import csv
import numpy    as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import cross_validation
from toolbox_02450 import feature_selector_lr, bmplot
from pylab import *

# DONT CHANGE
ELIM_COMM = 1 # eliminate communities
ELIM_ATTR = 2 # eliminate attributes
FILL_MEAN = 3 # fill missing with mean






titlestr=''
# FUNNY CONSTANTS HAHA
DISCARD     = 5 # discard first DISCARD columns
INVALID_FIX = ELIM_COMM
NORMALIZE   = True
STANDARDIZE = False
PRENORM     = False
REMOVE_N 	= 0


filestr = ''
# read data from comma-separated datafile
datafile = open('data/communities.data')
if PRENORM:
	datafile = open('data/communities_norm.data')
namefile = open('data/communities.names')
datareader = csv.reader(datafile, delimiter=',', quotechar='|')
namereader = csv.reader(namefile, delimiter=',', quotechar='|')

data = [[ np.nan if cell is '?' else float(cell)
          for cell in row[DISCARD:]] for row in datareader]


# attributeNames: a Mx1 matrix
attributeNames = namereader.next()[DISCARD:] # only one row in file so pop from iterable

# X: data matrix, rows correspond to N data objects, each of which contains M attributes
X = ma.masked_invalid(data) # missing values are masked

X_mean = X.mean(0)[np.newaxis,:] # every value is the attributes mean

# preprocesses X
if INVALID_FIX is ELIM_COMM:
	# removes data objects with missing values
	X = ma.compress_rows(X)
	titlestr += "Rows with missings removed. "
	filestr += 'rows-with-missings-removed_'
elif INVALID_FIX is ELIM_ATTR:
	# removes attributes with missing values
	X = ma.compress_cols(X)
	titlestr += "Attributes with missings removed. "
	filestr += 'attr-with-missings-removed_'
elif INVALID_FIX is FILL_MEAN:
	# takes mean where masked value otherwise
	X = np.where(X.mask, X_mean, X)
	titlestr += "Missings filled with means. "
	filestr += "missings-filled-w-means_"

X_mean = X.mean(0)[np.newaxis,:] # recalculate mean

removes = [4,16,7,210,247]

for i in range(REMOVE_N):
	X = np.delete(X, (removes[i]), axis=0)

titlestr += str(REMOVE_N) + " biggest outliers removed"
filestr += str(REMOVE_N)+"-biggest-outliers-removed_"

#X = np.delete(X, (119), axis=0) # remove row
#X = np.delete(X, (4), axis=0) # remove row
#X = np.delete(X, (7), axis=0) # remove row
#X = np.delete(X, (1432), axis=0) # remove row

# boxplot of each attribute
#figure()
#title('Boxplot of attributes')
#boxplot(X)
#xticks(range(len(attributeNames)), attributeNames, rotation=45)

# N: number of data objects
# M: number of attributes
(N,M) = X.shape

# OTHER VARIABLE NAMES:
# y: class index, a (Nx1) matrix.
#      for each data object, y contains a class index, y in {0,1,...,C-1}
#      where C is number of classes
# classNames:     a Cx1 matrix
# C:              number of classes




X_unNorm = np.copy(X);
X_unNorm_mean = np.copy(X_mean)
if NORMALIZE:
    titlestr += 'Normalized. '
    filestr += 'norm_'
    attributeSums = X.max(axis=1)
    X = X / attributeSums[:, np.newaxis] # normalized
    X_mean = X.mean(0)[np.newaxis,:] # mean along attributes, all valuesa are attr-mean

if STANDARDIZE:
    titlestr += 'standardized. '
    filestr += 'standrd_'
    X = (X - X_mean) / std(X)
    X_mean = X.mean(0)[np.newaxis,:] # mean along attributes, all valuesa are attr-mean

if PRENORM:
    titlestr += 'prenormed. '
    filestr  += 'prenormed_'


## PeeCeeAaa
# subtracts mean
Y = X - X_mean # last part adds dimension to array
Y_u = X_unNorm - X_unNorm_mean


# computes PCA, by computing SVD of Y
# u contains eigenvectors
U,S,V = linalg.svd(Y,full_matrices=False)
U_u,S_u,V_u = linalg.svd(Y_u,full_matrices=False)

print("index of biggest component of first eigenvector")
i1 = (U[1,:].argmax())

print("index of biggest component of second eigenvector")
i2 = (U[2,:].argmax())




#for i in range(M):
#	for j in range(i+1,M):
#		figure()
#		scatter(X[:,i], X[:,j])

#		tt = attributeNames[j] + " as a function of " + attributeNames[i]
#		title(tt)
#		xlabel(attributeNames[i])
#		ylabel(attributeNames[j])
#		savefig("correlations/"+attributeNames[j] + "-as-func-of-" + attributeNames[i]+".png")



# computes variance explained by principal components
rho = (S*S) / (S*S).sum() 
cumrho = cumsum(rho)

# plot it
#fig_pca = figure()

#ax = fig_pca.add_subplot(1,1,1)
#ax.plot(cumrho)
#ax.set_title("Variance as a function of number of PCs")
#ax.set_xlabel("Number of principal components included")
#ax.set_ylabel("Amount of variance explained")

#fig_pca.savefig('variance'+filestr+'.png')

# projects the centered data onto principal component space, Z
V = mat(V).T
V_u = mat(V_u).T
Z = Y * V
Z_u = Y_u * V_u


maxindex = Z.T.argmax()
print(maxindex)





# Indices of the principal components to be plotted
i = 0
j = 1
k = -5
print(attributeNames[k])

x = Z[:,i].flat
x_u = Z_u[:,i].flat
y = Z[:,j].flat
y_u = Z_u[:,j].flat
z = X_unNorm[:,-1].flat
zn = X[:,-1].flat
print(zn)

zmax = (X_unNorm[:,-1].max())
zmin = (X_unNorm[:,-1].min())


# Plot PCA of the data
#cm = plt.cm.hot

#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#ax1.scatter(x,y,c=z, label="lol", cmap=cm, vmin=zmin, vmax=zmax)
#ax1.set_title("PCA: " + titlestr)
#ax1.set_xlabel("PCA #"+str(i))
#ax1.set_ylabel("PCA #"+str(j))
#savefig("pca/"+filestr+".png")

#plt.show()


#cax = ax.imshow(data, interpolation='nearest', vmin=0.5, vmax=0.99)
#fig.colorbar(c)

#figure()
#title('PCAAAAAAAAAAAAA')
#plot(Z[:,i], Z[:,j], 'o')
#axis('equal')
# Output result to screen
#show()


#regression

# Split dataset into features and target vector
autotheft_idx = attributeNames.index('autoTheftPerPop')
y = X[:, autotheft_idx]

X_cols = range(0, autotheft_idx) + range(autotheft_idx + 1, len(attributeNames))
X_rows = range(0, len(y))
X = X[ix_(X_rows, X_cols)]

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

show()

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = cross_validation.KFold(N,K,shuffle=True)

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


show()    