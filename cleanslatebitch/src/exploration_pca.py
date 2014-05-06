import pylab as pl


from Framework.DataSet import *
from Framework.PCA import *
from Tools import writeapriorifile

dataset = DataSet(
	datafile ='../data/normalized.csv',
	na_values=['?'],
	string_columns=['state','communityname'],
)


#dataset = dataset.drop_rows([21])

#dataset = dataset.standardize()
dataset = dataset.fix_missing(drop_attributes=True)	
#print(dataset.df)

pca = PCA(dataset)

pca.plot(color='medIncome')