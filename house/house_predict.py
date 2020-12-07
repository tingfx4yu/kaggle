import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
house = pd.read_csv('house/train.csv')
test = pd.read_csv('house/test.csv')
all_data = pd.concat([house,test],ignore_index = True)
#MainFeature :?BsmtQual, YearBuilt , CentralAir, OverallQual, ?BsmtFinType1[GLQ:1,ow:0]
#?HeatQC[Ex:1,ow:0],'GarageArea','GarageCars', GrLivArea


#Final feature:
# OverallQual,GrLivArea,GarageCars,TotalBsmtSF,
# FullBath,YearBulit, 1stFlrSF, TotRmsAbvGrd


########## Built Year ################ #1872 -- 2010
def YearProcessing(x):
    if 1870 < x < 1920:
        return 1
    elif 1920<= x < 1950:
        return 2
    elif 1950<= x < 1970:
        return 3
    elif 1970 <= x < 1990:
        return 4
    elif 1990 <= x < 2011:
        return 5
YearDf = pd.DataFrame()
YearDf = all_data['YearBuilt'].map(YearProcessing)
#print(len(YearDf))

########## GrLivArea #######  334 -- 5642
def LiveProcessing(x):
    if x > 4500:
        return 750000
    else:
        return x

GrLivAreaDf = all_data['GrLivArea'].map(LiveProcessing)


###############GarageCars##############
GarageCarsDf = all_data['GarageCars'].fillna(3.0)


##############OverallQual##############


##############TotalBsmtSF###############

TotalBsmtSFDf = all_data['TotalBsmtSF'].fillna(1051.1)

#TotalBsmtSFDf.fillnull(1051.1)


############FullBath################

def FullBathProcessing(x):
    if x >= 14:
        return 400000
    else:
        return x

FullBathDf = all_data['FullBath'].map(FullBathProcessing)


###############1stFlrSF######################

def _1stFlrSFProcessing(x):
    if x >= 2500:
        return 450000
    else:
        return x

_1stFlrSFDf = all_data['1stFlrSF'].map(_1stFlrSFProcessing)


#################################################
handled = pd.concat([YearDf,GrLivAreaDf,GarageCarsDf,TotalBsmtSFDf,all_data['OverallQual'],FullBathDf,_1stFlrSFDf],axis= 1)
#print(handled.info())

SourceRow = 1460
sourceX = handled.loc[0:SourceRow-1,:]
sourceY = all_data.loc[0:SourceRow-1,'SalePrice']
PredX = handled.loc[SourceRow:,:]
trainX,testX,trainY,testY = train_test_split(sourceX,sourceY,train_size =0.77,random_state = 42)
model = GradientBoostingRegressor(n_estimators=100,subsample=0.8,loss='huber')
model.fit(trainX,trainY)
print(model.score(testX,testY))
predY = model.predict(PredX)
Id = all_data.loc[SourceRow:,'Id']
predDf = pd.DataFrame({'Id':Id,'SalePrice':predY})

print(predDf.head(20))

predDf.to_csv('gbdtres.csv',index = False)