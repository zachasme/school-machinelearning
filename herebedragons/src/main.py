import csv
import numpy as np

DATADIR = '../data/'

"""read attribute data and names from comma-separated files"""
namefile = open(DATADIR + 'communities.names')
datafile = open(DATADIR + 'communities.data')
#datafile = open('data/communities_norm.data')

datareader = csv.reader(datafile, delimiter=',', quotechar='|')
namereader = csv.reader(namefile, delimiter=',', quotechar='|')

data = []
for rowreader in datareader:
	row = []
	for cell in rowreader:
		try:
			row.append(float(cell))
		except:
			row.append(np.nan)
	data.append(row)

names = next(namereader)
drop_columns = [
	'State', 'countyCode', 'communityCode', 'communityname',
	'murders', 'murdPerPop',
	'rapes', 'rapesPerPop',
	'robberies', 'robbbPerPop',
	'assaults', 'assaultPerPop',
	'burglaries', 'burglPerPop',
	'larcenies', 'larcPerPop',
#	'autoTheft', 'autoTheftPerPop',
	'arsons', 'arsonsPerPop',
	'violentPerPop',
	'nonViolPerPop',
]






from DataSet import *
from PCA import *

dataset = DataSet(data, names, drop_columns=drop_columns, fix_missing=FixMissing.DROPATTRIBUTES, rescale=Rescale.NORMALIZE)
print(dataset.X)

print(dataset.X.iloc[5,10])

pca = PCA(dataset)
pca.plot_rho()
pca.show()
plt.show()

print("\n\nstd:",   dataset.X.std())
print("\n\nmean:",  dataset.X.mean())
print("\n\nrange:", dataset.X.max()-dataset.X.min())