import csv

DISCARD = 5 # discard first DISCARD columns
DATADIR = '../data/'

"""read attribute data and names from comma-separated files"""
namefile = open(DATADIR + 'communities.names')
datafile = open(DATADIR + 'communities.data')
#datafile = open('data/communities_norm.data')

datareader = csv.reader(datafile, delimiter=',', quotechar='|')
namereader = csv.reader(namefile, delimiter=',', quotechar='|')

data = [[ '?' if cell is '?' else float(cell)
          for cell in row[DISCARD:]] for row in datareader]





import Oli

oli = Oli.Oli(data)

Oli.PCA(oli)