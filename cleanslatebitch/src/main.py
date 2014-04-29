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
<<<<<<< HEAD
print(crime.df.arsons)
crime = crime.classIn('arsons')
print(crime)
=======

crime2 = crime.binarize('arsons')
print(crime2);

crime3 = crime.classIn('arsons')
print(crime3.y);
>>>>>>> a0e3d83a2d286e76dc97dd0138b0fde023aa60fa


crime = crime.binarize('state')
#print(crime)
#from Framework.PCA import *
#pca = PCA(crime)
#pca.plot()
