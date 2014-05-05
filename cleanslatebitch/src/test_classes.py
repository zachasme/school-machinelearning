import pylab as pl


from Framework.DataSet import *

dataset = DataSet(
	datafile ='../data/raw.csv',
	na_values=['?'],
	string_columns=['state','communityname'],
)

dataset = dataset.fix_missing(drop_objects=True)
dataset = dataset.standardize()
dataset = dataset.discretize('arsons', 2);
dataset = dataset.set_class_column('arsons', nodelete=True)

print ( dataset.y )


#dataset = dataset.normalize()

