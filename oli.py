import csv
import numpy    as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from pylab import *

# DONT CHANGE
ELIM_COMM = 1 # eliminate communities
ELIM_ATTR = 2 # eliminate attributes
FILL_MEAN = 3 # fill missing with mean







# FUNNY CONSTANTS HAHA
DISCARD     = 5 # discard first DISCARD columns
INVALID_FIX = ELIM_COMM
NORMALIZE   = False
STANDARDIZE = True

# read data from comma-separated datafile
datafile = open('data/communities.data')
namefile = open('data/communities.names')
datareader = csv.reader(datafile, delimiter=',', quotechar='|')
namereader = csv.reader(namefile, delimiter=',', quotechar='|')

data = [[ np.nan if cell is '?' else float(cell)
          for cell in row[DISCARD:]] for row in datareader]

# X: data matrix, rows correspond to N data objects, each of which contains M attributes
X = ma.masked_invalid(data) # missing values are masked

X_mean = X.mean(0)[np.newaxis,:] # every value is the attributes mean

# preprocesses X
if INVALID_FIX is ELIM_COMM:
    # removes data objects with missing values
    X = ma.compress_rows(X)
elif INVALID_FIX is ELIM_ATTR:
    # removes attributes with missing values
    X = ma.compress_cols(X)
elif INVALID_FIX is FILL_MEAN:
    # takes mean where masked value otherwise
    X = np.where(X.mask, X_mean, X)

X_mean = X.mean(0)[np.newaxis,:] # recalculate mean

#X = np.delete(X, (21), axis=0) # remove row
#X = np.delete(X, (126), axis=0) # remove row
#X = np.delete(X, (1432), axis=0) # remove row



# M: number of attributes
# N: number of data objects
(M,N) = X.shape
# attributeNames: a Mx1 matrix
attributeNames = namereader.next() # only one row in file so pop from iterable

# OTHER VARIABLE NAMES:
# y: class index, a (Nx1) matrix.
#      for each data object, y contains a class index, y in {0,1,...,C-1}
#      where C is number of classes
# classNames:     a Cx1 matrix
# C:              number of classes




X_unNorm = np.copy(X);
X_unNorm_mean = np.copy(X_mean)
if NORMALIZE:
    attributeSums = X.max(axis=1)
    X = X / attributeSums[:, np.newaxis] # normalized
    X_mean = X.mean(0)[np.newaxis,:] # mean along attributes, all valuesa are attr-mean

if STANDARDIZE:
    X = (X - X_mean) / std(X)
    X_mean = X.mean(0)[np.newaxis,:] # mean along attributes, all valuesa are attr-mean




## PeeCeeAaa
# subtracts mean
Y = X - X_mean # last part adds dimension to array
Y_u = X_unNorm - X_unNorm_mean


# computes PCA, by computing SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)
U_u,S_u,V_u = linalg.svd(Y_u,full_matrices=False)

# computes variance explained by principal components
rho = (S*S) / (S*S).sum() 
cumrho = cumsum(rho)

# plot it
fig_pca = figure()

ax = fig_pca.add_subplot(1,1,1)
ax.plot(cumrho)
ax.set_title("Variance as a function of number of PCs")
ax.set_xlabel("Number of principal components included")
ax.set_ylabel("Amount of variance explained")

fig_pca.savefig('pca.png')

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
cm = plt.cm.hot

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.scatter(x_u,y_u,c=z, label="lol", cmap=cm, vmin=zmin, vmax=zmax)
ax1.set_title("Communities in PCA space, color is violent crime per capita")
ax1.set_xlabel("PCA #"+str(i))
ax1.set_ylabel("PCA #"+str(j))


ax2.scatter(x,y,c=zn, label="lol", cmap=cm)
ax2.set_title("Communities in PCA space, color is NORM violent crime per capita")
ax2.set_xlabel("PCA #"+str(i))
ax2.set_ylabel("PCA #"+str(j))


plt.show()


#cax = ax.imshow(data, interpolation='nearest', vmin=0.5, vmax=0.99)
#fig.colorbar(c)

#figure()
#title('PCAAAAAAAAAAAAA')
#plot(Z[:,i], Z[:,j], 'o')
#axis('equal')
# Output result to screen
#show()
