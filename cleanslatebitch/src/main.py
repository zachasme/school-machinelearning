import pylab as pl


from Framework.DataSet import *

dataset = DataSet(
	datafile ='../data/raw.csv',
	na_values=['?'],
	string_columns=['state','communityname'],
)

dataset = dataset.standardize()
dataset = dataset.take_columns(['arsons'])

print(dataset.df)

#print(type("wat"))

dataset = dataset.fix_missing(drop_objects=True)
dataset = dataset.standardize()



#dataset = dataset.normalize()

