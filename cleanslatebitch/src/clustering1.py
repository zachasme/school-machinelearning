from Framework.DataSet import *

from pylab import *
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

crime = DataSet(datafile='../data/normalized.csv', string_columns=['state','communityname','countyCode','communityCode'])

#crime = crime.drop(['state', 'communityname']) 	  # Drop strings
#crime = crime.drop(['countyCode','communityCode']) # Drop nominals
# crime = crime.drop_columns([
# 	'fold',
# 	'murders', 'murdPerPop',
# 	'rapes', 'rapesPerPop',
# 	'robberies', 'robbbPerPop',
# #	'assaults', 'assaultPerPop',
# 	'burglaries', 'burglPerPop',
# 	'larcenies', 'larcPerPop',
# 	'autoTheft', 'autoTheftPerPop',
# 	'arsons', 'arsonsPerPop',
# 	'ViolentCrimesPerPop',
# 	'nonViolPerPop',
# ])
crime = crime.take_columns([
	'racePctHisp', 'racePctWhite',# 'racepctblack',
	'medIncome', 'NumStreet', 'NumUnderPov'#, 'NumImmig', 'NumInShelters',
	#'PctEmploy'
	])
#crime = crime.fix_missing(fill_mean=True)
#crime = crime.standardize()
#crime = crime.normalize()
#crime = crime.drop_nominals()
#crime = crime.discretize('assaults', 3)
crime = DataSet(dataframe=crime.df[:100])
#print(crime.df.assaults)
print(crime.attributeNames)

# Variables of interest
N, M = crime.N, crime.M
#C = len(crime.classNames)
X = crime.X

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'single'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 4
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
#clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2)
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()