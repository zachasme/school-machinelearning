# exercise 10.1.1

from pylab import *

from scipy.io import loadmat

from toolbox_02450 import clusterplot

from sklearn.mixture import GMM



# Load Matlab data file and extract variables of interest
from Framework.DataSet import *

crime = DataSet(
	datafile ='../data/raw.csv',
	na_values=['?'],
	nominals =['communityname','countyCode','communityCode'],
	class_column = 'state'
)

crime = crime.drop_columns([
	'fold',
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

data = crime

data = data.fix_missing(drop_objects=True)
data = data.drop_nominals()
data = data.normalize()

#mat_data = loadmat('../Data/synth1.mat')
#X = np.matrix(mat_data['X'])
X = data.X
#y = np.matrix(mat_data['y'])
y = data.y
#attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
attributeNames = data.attributeNames
#classNames = [name[0][0] for name in mat_data['classNames']]
classNames = data.classNames
N, M = X.shape
C = len(classNames)




# Number of clusters

K = 4

cov_type = 'diag'       # type of covariance, you can try out 'diag' as well

reps = 1                # number of fits with different initalizations, best result will be kept



# Fit Gaussian mixture model

gmm = GMM(n_components=K, covariance_type=cov_type, n_init=reps, params='wmc').fit(X)

cls = gmm.predict(X)    # extract cluster labels

cds = gmm.means_        # extract cluster centroids (means of gaussians)

covs = gmm.covars_      # extract cluster shapes (covariances of gaussians)



if cov_type == 'diag':

    new_covs = np.zeros([K,M,M])

    count = 0

    for elem in covs:

        temp_m = np.zeros([M,M])

        for i in range(len(elem)):

            temp_m[i][i] = elem[i]

        new_covs[count] = temp_m

        count += 1

    covs = new_covs



# Plot results:

figure(figsize=(14,9))

#clusterplot(X[1:2], clusterid=cls, centroids=cds, y=y, covars=covs)

show()