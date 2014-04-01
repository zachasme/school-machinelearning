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
			if cell == '?':
				row.append(np.nan)
			else:
				row.append(cell)
	data.append(row)

names = next(namereader)
drop_columns = [
	'State',
	'countyCode', 'communityCode', 'communityname',
	'fold',
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

import matplotlib.pyplot as plt

dataset = DataSet(data, names, class_column='State', drop_columns=drop_columns, fix_missing=FixMissing.DROPOBJECTS, rescale=Rescale.NORMALIZE)

print (dataset.X)
print ("M" + str(dataset.M))
print ("N" + str(dataset.N))
print ("C" + str(dataset.C))

print(len(dataset.y))