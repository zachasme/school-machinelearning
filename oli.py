import csv
import numpy    as np
import numpy.ma as ma

from pylab import *

# DONT CHANGE
ELIM_COMM = 1 # eliminate communities
ELIM_ATTR = 2 # eliminate attributes
FILL_MEAN = 3 # fill missing with mean







# FUNNY CONSTANTS HAHA
DISCARD = 5 # discard first DISCARD columns
INVALID_FIX = ELIM_ATTR

# read data from comma-separated datafile
datafile = open('data/communities.data')
datareader = csv.reader(datafile, delimiter=',', quotechar='|')

data = [[ np.nan if cell is '?' else float(cell)
          for cell in row[DISCARD:]] for row in datareader]

# X: data matrix, rows correspond to N data objects, each of which contains M attributes
X = ma.masked_invalid(data) # missing values are masked

# preprocesses X
if INVALID_FIX is ELIM_COMM:
    # removes data objects with missing values
    X = ma.compress_rows(X)
elif INVALID_FIX is ELIM_ATTR:
    # removes attributes with missing values
    X = ma.compress_cols(X)
elif INVALID_FIX is FILL_MEAN:
    # takes mean where masked value otherwise
    X = np.where(X.mask, mean[np.newaxis,:], X)

#X = np.delete(X, (21), axis=0) # remove row
#X = np.delete(X, (126), axis=0) # remove row
#X = np.delete(X, (1432), axis=0) # remove row

attributeSums = X.sum(axis=1)
X = X / attributeSums[:, np.newaxis]

# M: number of attributes
# N: number of data objects
(M,N) = X.shape

print(N)
# OTHER VARIABLE NAMES:
# y: class index, a (Nx1) matrix.
#      for each data object, y contains a class index, y in {0,1,...,C-1}
#      where C is number of classes
# attributeNames: a Mx1 matrix
# classNames:     a Cx1 matrix
# C:              number of classes

## PeeCeeAaa
# subtracts mean
print(X)

mean = X.mean(0)
Y = X - mean[np.newaxis,:]


# computes PCA, by computing SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)

# computes variance explained by principal components
rho = (S*S) / (S*S).sum() 
#print(rho)

# projects the centered data onto principal component space, Z
V = mat(V).T
Z = Y * V


maxindex = Z.T.argmax()
print(maxindex)



# Indices of the principal components to be plotted
i = 0
j = 1


# Plot PCA of the data
figure()
title('PCAAAAAAAAAAAAA')
plot(Z[:,i], Z[:,j], 'o')
axis('equal')
# Output result to screen
show()
