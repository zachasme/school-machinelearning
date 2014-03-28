import csv
import numpy as np

PREDISCARD = 5 # discard first DISCARD columns
DISCARD = 5 # discard first DISCARD columns
DATADIR = '../data/'

"""read attribute data and names from comma-separated files"""
namefile = open(DATADIR + 'communities.names')
datafile = open(DATADIR + 'communities.data')
#datafile = open('data/communities_norm.data')

datareader = csv.reader(datafile, delimiter=',', quotechar='|')
namereader = csv.reader(namefile, delimiter=',', quotechar='|')

data = [[ np.nan if cell is '?' else float(cell)
          for cell in row[DISCARD:]] for row in datareader]
names = next(namereader)




from Oli import *

oli = DataSet(data, names[DISCARD:], fixna=Fixna.FILLMEAN, standardize=True)

print(oli.X.std())
print(oli.X.mean())

#Oli.PCA(oli)