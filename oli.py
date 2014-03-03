import csv
import numpy    as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from pylab import *

# DONT CHANGE
ELIM_COMM = 1 # eliminate communities
ELIM_ATTR = 2 # eliminate attributes
FILL_MEAN = 3 # fill missing with mean






titlestr=''
# FUNNY CONSTANTS HAHA
DISCARD     = 5 # discard first DISCARD columns
INVALID_FIX = FILL_MEAN
NORMALIZE   = False
STANDARDIZE = False
PRENORM     = True
REMOVE_N = 0


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
fig_pca = figure()

ax = fig_pca.add_subplot(1,1,1)
ax.plot(cumrho)
ax.set_title("Variance as a function of number of PCs")
ax.set_xlabel("Number of principal components included")
ax.set_ylabel("Amount of variance explained")

fig_pca.savefig('variance'+filestr+'.png')

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
ax1 = fig.add_subplot(1,1,1)
ax1.scatter(x,y,c=z, label="lol", cmap=cm, vmin=zmin, vmax=zmax)
ax1.set_title("PCA: " + titlestr)
ax1.set_xlabel("PCA #"+str(i))
ax1.set_ylabel("PCA #"+str(j))
savefig("pca/"+filestr+".png")

plt.show()


#cax = ax.imshow(data, interpolation='nearest', vmin=0.5, vmax=0.99)
#fig.colorbar(c)

#figure()
#title('PCAAAAAAAAAAAAA')
#plot(Z[:,i], Z[:,j], 'o')
#axis('equal')
# Output result to screen
#show()
