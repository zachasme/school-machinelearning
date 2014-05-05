import pylab as pl


from Framework.DataSet import *

dataset = DataSet(
	datafile ='../data/normalized.csv',
	na_values=['?'],
	string_columns=['state','communityname'],
)


dataset = dataset.fix_missing(drop_objects=True)
dataset = dataset.standardize()

dataset = dataset.discretize('state', 2);

dataset = dataset.set_class_column('state')

print ( dataset.y )


#dataset = dataset.normalize()

