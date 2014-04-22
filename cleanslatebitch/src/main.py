from DataSet import *

datafile = '../data/raw.csv'
crime    = DataSet(datafile)

crime.drop([
	'state',
#	'countyCode',
#	'communityCode',
	'communityname',
#	'fold',
#	'murders', 'murdPerPop',
#	'rapes', 'rapesPerPop',
#	'robberies', 'robbbPerPop',
#	'assaults', 'assaultPerPop',
#	'burglaries', 'burglPerPop',
#	'larcenies', 'larcPerPop',
#	'autoTheft', 'autoTheftPerPop',
#	'arsons', 'arsonsPerPop',
#	'ViolentPerPop',
#	'nonViolPerPop',
])


print(crime.binarize('assaults', 4))