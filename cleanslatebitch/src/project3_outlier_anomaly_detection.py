import pylab as pl


from Framework.DataSet import *
from Tools import writeapriorifile

dataset = DataSet(
	datafile ='../data/normalized.csv',
	na_values=['?'],
	string_columns=['state','communityname'],
)

dataset = dataset.drop_columns([
    'state',
#    'communityname',
	'fold',
#	'murders', 'murdPerPop',
#	'rapes', 'rapesPerPop',
#	'robberies', 'robbbPerPop',
#	'assaults', 'assaultPerPop',
#	'burglaries', 'burglPerPop',
#	'larcenies', 'larcPerPop',
#	'autoTheft', 'autoTheftPerPop',
#	'arsons', 'arsonsPerPop',
#	'ViolentCrimesPerPop',
#	'nonViolPerPop',
])
#dataset = dataset.standardize()

dataset = dataset.standardize();

dataset = dataset.fix_missing(drop_objects=True)

dataorig = dataset;

dataset = dataset.take_rows(range(1000));

#dataset = dataset.discretize('arsons', 2)



X = dataset.X
M = dataset.M
N = dataset.N


displayN = 50




from pylab import *
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors





### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X,width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i]


# Display the index of the lowest density data object
print('Gauss Kernel: Lowest density: {0} for data object: {1}'.format(density[0,0],i[0]))


# Plot density estimate
figure(1)
bar(range(displayN),density[:displayN])
title('Gaussian Kernel Density estimate')






### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i = density.argsort()
density = density[i]


# Display the index of the lowest density data object
print('K neigb density: Lowest density: {0} for data object: {1}'.format(density[0],i[0]))


# Plot k-neighbor density estimate 
figure(3)
bar(range(displayN),density[:displayN])
title('KNN density: Outlier score')








### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]


# Display the index of the lowest density data object
print('KNN avg rel: {0} for data object: {1}'.format(avg_rel_density[0],i_avg_rel[0]))


# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(displayN),avg_rel_density[:displayN])
title('KNN average relative density: Outlier score')







### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Outlier score
score = D[:,K-1]

# Sort the scores
i = score.argsort()
i = i[::-1]
score = score[i]


# Display the index of the highest score data object
print('5th nearest neighb: highest distance {0} for data object: {1}'.format(score[0],i[0]))


# Plot k-neighbor estimate of outlier score (distances)
figure(7)
bar(range(displayN),score[:displayN])
title('5th neighbor distance')


show()
