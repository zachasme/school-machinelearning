import csv
import numpy as np

from pylab import *

attributeNames = [
'communityname','State','countyCode','communityCode','fold','pop','perHoush','pctBlack','pctWhite','pctAsian','pctHisp','pct12-21','pct12-29','pct16-24','pct65up','persUrban','pctUrban','medIncome','pctWwage','pctWfarm','pctWdiv','pctWsocsec','pctPubAsst','pctRetire','medFamIncome','perCapInc','whitePerCap','blackPerCap','NAperCap','asianPerCap','otherPerCap','hispPerCap','persPoverty','pctPoverty','pctLowEdu','pctNotHSgrad','pctCollGrad','pctUnemploy','pctEmploy','pctEmployMfg','pctEmployProfServ','pctOccupManu','pctOccupMgmt','pctMaleDivorc','pctMaleNevMar','pctFemDivorc','pctAllDivorc','persPerFam','pct2Par','pctKids2Par','pctKids-4w2Par','pct12-17w2Par','pctWorkMom-6','pctWorkMom-18','kidsBornNevrMarr','pctKidsBornNevrMarr','numForeignBorn','pctFgnImmig-3','pctFgnImmig-5','pctFgnImmig-8','pctFgnImmig-10','pctImmig-3','pctImmig-5','pctImmig-8','pctImmig-10','pctSpeakOnlyEng','pctNotSpeakEng','pctLargHousFam','pctLargHous','persPerOccupHous','persPerOwnOccup','persPerRenterOccup','pctPersOwnOccup','pctPopDenseHous','pctSmallHousUnits','medNumBedrm','houseVacant','pctHousOccup','pctHousOwnerOccup','pctVacantBoarded','pctVacant6up','medYrHousBuilt','pctHousWOphone','pctHousWOplumb','ownHousLowQ','ownHousMed','ownHousUperQ','ownHousQrange','rentLowQ','rentMed','rentUpperQ','rentQrange','medGrossRent','medRentpctHousInc','medOwnCostpct','medOwnCostPctWO','persEmergShelt','persHomeless','pctForeignBorn','pctBornStateResid','pctSameHouse-5','pctSameCounty-5','pctSameState-5','numPolice','policePerPop','policeField','policeFieldPerPop','policeCalls','policCallPerPop','policCallPerOffic','policePerPop2','racialMatch','pctPolicWhite','pctPolicBlack','pctPolicHisp','pctPolicAsian','pctPolicMinority','officDrugUnits','numDiffDrugsSeiz','policAveOT','landArea','popDensity','pctUsePubTrans','policCarsAvail','policOperBudget','pctPolicPatrol','gangUnit','pctOfficDrugUnit','policBudgetPerPop','murders','murdPerPop','rapes','rapesPerPop','robberies','robbbPerPop','assaults','assaultPerPop','burglaries','burglPerPop','larcenies','larcPerPop','autoTheft','autoTheftPerPop','arsons','arsonsPerPop','violentPerPop','nonViolPerPop',
]

csvfile = open('data/communities.data')
csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
data = list(csvreader)
M = len(attributeNames)-5
N = len(data)

X = np.zeros(shape=(N,M))

for idx, row in enumerate(data):
    while '?' in row:
        row[row.index('?')] = -1
    X[idx,:] = row[5:]

## P C fucking A
#subtract mean
Y = X - np.ones((N,1))*X.mean(0)
#PCA by computing SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)


# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 


V = mat(V).T

# Project the centered data onto principal component space
Z = Y * V


# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
figure()
title('NanoNose data: PCA')
plot(Z[:,i], Z[:,j], 'o')

# Output result to screen
show()