import pylab as pl


from Framework.DataSet import *

crime = DataSet(datafile='../data/raw.csv', nominals=['state','communityname','countyCode','communityCode'])

crime = crime.drop_columns([
	'fold',
	'murders', 'murdPerPop',
	'rapes', 'rapesPerPop',
	'robberies', 'robbbPerPop',
#	'assaults', 'assaultPerPop',
	'burglaries', 'burglPerPop',
	'larcenies', 'larcPerPop',
	'autoTheft', 'autoTheftPerPop',
#	'arsons', 'arsonsPerPop',
	'ViolentCrimesPerPop',
	'nonViolPerPop',
])
crime = crime.normalize()


crime = crime.discretize('arsons', 4)
print(crime.df.arsons)
crime = crime.classIn('arsons')
print(crime)


crime = crime.binarize('state')
#print(crime)
#from Framework.PCA import *
#pca = PCA(crime)
#pca.plot()
