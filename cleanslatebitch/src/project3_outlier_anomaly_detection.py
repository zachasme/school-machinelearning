from Framework.DataSet import *
from Tools import writeapriorifile
import pylab as pl

dataset = DataSet(
	datafile ='../data/raw.csv',
	na_values=['?'],
	string_columns=['state'],
)
dataset = dataset.set_class_column('communityname')

dataset = dataset.drop_columns([
#   'communityname',
#	'countyCode',
##	'communityCode',
#	'fold',
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

dataset = dataset.fix_missing(drop_attributes=True)



outer_n = 5
inner_n = 3
for outer_i in range(outer_n):
	
	
	X = dataset.X
	M = dataset.M
	N = dataset.N
	
	
	displayN = 10
	
	
	
	
	
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
	density, og_density = gausKernelDensity(X,width)
	
	# Sort the densities
	i = (density.argsort(axis=0)).ravel()
	density = density[i]
	
	
	# Display the index of the lowest density data object
	for inner_i in range(inner_n):
		print('Gauss Kernel: {2}. lowest density: {0} for data object: {1}'.format(density[inner_i,0],i[inner_i],inner_i+1))
	
	candidate_gauss = i[0]

	
	# Plot density estimate
	figure(1)
	bar(range(displayN),density[:displayN])
	title('Gaussian Kernel Density estimate')
	ylabel('Density Estimate')
	xlabel('Object index')
	xticks(range(displayN), i[:displayN])
	
	
	
	
	
	
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
	for inner_i in range(inner_n):
		print('K neigb density: {2}. lowest density: {0} for data object: {1}'.format(density[inner_i],i[inner_i],inner_i+1))

	candidate_kn = i[0]	
	
	# Plot k-neighbor density estimate 
	figure(3)
	bar(range(displayN),density[:displayN])
	title('K-Nearest Neighbours density estimate')
	ylabel('Density Estimate')
	xlabel('Object index')
	xticks(range(displayN), i[:displayN])
	
	
	
	
	
	
	
	
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
	for inner_i in range(inner_n):
		print('KNN avg rel: {2}. worst is {0} for data object: {1}'.format(avg_rel_density[inner_i],i_avg_rel[inner_i],inner_i+1))
	
	candidate_knavg = i_avg_rel[0]
	
	# Plot k-neighbor estimate of outlier score (distances)
	figure(5)
	bar(range(displayN),avg_rel_density[:displayN])
	title('KNN average relative density estimate')
	ylabel('Density Estimate')
	xlabel('Object index')
	xticks(range(displayN), i_avg_rel[:displayN])
	
	
	
	
	
	
	
	
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
	for inner_i in range(inner_n):
		print('5th nearest neighb: {2}. highest distance {0} for data object: {1}'.format(score[inner_i],i[inner_i],inner_i+1))
	
	candidate_kdist = i[0]
	
	# Plot k-neighbor estimate of outlier score (distances)
	figure(7)
	bar(range(displayN),score[:displayN])
	title('Kth nearest neighbor distance (K=5)')
	ylabel('Density Estimate')
	xlabel('Object index')
	xticks(range(displayN), i[:displayN])
	
	
	
	show()
	

	if candidate_kdist == candidate_knavg and candidate_kdist == candidate_kn and candidate_kdist == candidate_gauss:
		dataset = dataset.drop_rows([candidate_kdist])
	else:
		break