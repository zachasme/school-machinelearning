import csv
import numpy as np

from pylab import *

attributeNames = [
'communityname','State','countyCode','communityCode','fold','pop','perHoush','pctBlack','pctWhite','pctAsian','pctHisp','pct12-21','pct12-29','pct16-24','pct65up','persUrban','pctUrban','medIncome','pctWwage','pctWfarm','pctWdiv','pctWsocsec','pctPubAsst','pctRetire','medFamIncome','perCapInc','whitePerCap','blackPerCap','NAperCap','asianPerCap','otherPerCap','hispPerCap','persPoverty','pctPoverty','pctLowEdu','pctNotHSgrad','pctCollGrad','pctUnemploy','pctEmploy','pctEmployMfg','pctEmployProfServ','pctOccupManu','pctOccupMgmt','pctMaleDivorc','pctMaleNevMar','pctFemDivorc','pctAllDivorc','persPerFam','pct2Par','pctKids2Par','pctKids-4w2Par','pct12-17w2Par','pctWorkMom-6','pctWorkMom-18','kidsBornNevrMarr','pctKidsBornNevrMarr','numForeignBorn','pctFgnImmig-3','pctFgnImmig-5','pctFgnImmig-8','pctFgnImmig-10','pctImmig-3','pctImmig-5','pctImmig-8','pctImmig-10','pctSpeakOnlyEng','pctNotSpeakEng','pctLargHousFam','pctLargHous','persPerOccupHous','persPerOwnOccup','persPerRenterOccup','pctPersOwnOccup','pctPopDenseHous','pctSmallHousUnits','medNumBedrm','houseVacant','pctHousOccup','pctHousOwnerOccup','pctVacantBoarded','pctVacant6up','medYrHousBuilt','pctHousWOphone','pctHousWOplumb','ownHousLowQ','ownHousMed','ownHousUperQ','ownHousQrange','rentLowQ','rentMed','rentUpperQ','rentQrange','medGrossRent','medRentpctHousInc','medOwnCostpct','medOwnCostPctWO','persEmergShelt','persHomeless','pctForeignBorn','pctBornStateResid','pctSameHouse-5','pctSameCounty-5','pctSameState-5','numPolice','policePerPop','policeField','policeFieldPerPop','policeCalls','policCallPerPop','policCallPerOffic','policePerPop2','racialMatch','pctPolicWhite','pctPolicBlack','pctPolicHisp','pctPolicAsian','pctPolicMinority','officDrugUnits','numDiffDrugsSeiz','policAveOT','landArea','popDensity','pctUsePubTrans','policCarsAvail','policOperBudget','pctPolicPatrol','gangUnit','pctOfficDrugUnit','policBudgetPerPop','murders','murdPerPop','rapes','rapesPerPop','robberies','robbbPerPop','assaults','assaultPerPop','burglaries','burglPerPop','larcenies','larcPerPop','autoTheft','autoTheftPerPop','arsons','arsonsPerPop','violentPerPop','nonViolPerPop',
];

X = [];
killUs = [];

with open('data/communities.data') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    data = list(csvreader)
    X = np.zeros(shape=(len(data),len(attributeNames)-5))

    for idx, row in enumerate(data):
        if any(type(attr) is str for attr in row):
            print("oh fuck")
        else:
            X[idx] = row[5:]


#make plot
#figure()
#plot(X[:,1], X[:,2], 'o')
#show()

print(X)
print(attributeNames)