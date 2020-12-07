import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
house = pd.read_csv('house/train.csv')
test = pd.read_csv('house/test.csv')
print(len(house))
print(len(test))
all_data = pd.concat([house,test],ignore_index = True)
#print(all_data['TotalBsmtSF'].describe()) 
#print(all_data.head(10))
#print(all_data.info())
#MainFeature :BsmtQual, YearBuilt , CentralAir, OverallQual, BsmtFinType1[GLQ:1,ow:0]
#HeatQC[Ex:1,ow:0], MSZoning,'GarageArea','GarageCars',GrLivArea
def CheckRelationToPrice_Box(var):
    for i in range(len(var)): 
        data = pd.concat([all_data[var[i]],all_data.SalePrice],axis= 1)
        fig = sns.boxplot(x = var[i], y = 'SalePrice', data = all_data)
        fig.axis(ymin = 0,ymax = 800000)

def CheckRelationToPrice_Scatter(var):
    for i in range(len(var)):
        data = pd.concat([all_data[var[i]],all_data.SalePrice],axis= 1)
        data.plot.scatter(x = var[i],y = 'SalePrice', ylim = (0,800000))   
def HeatMap():
    f_name = ['CentralAir', 'Neighborhood']
    for x in f_name:
        label = preprocessing.LabelEncoder()
        house[x] = label.fit_transform(house[x])
    corrmat = house.corr()
    cols = corrmat.nlargest(10,'SalePrice')['SalePrice'].index
    #print(cols)
    cm = np.corrcoef(house[cols].values.T)
    #print(cm)
    sns.set(font_scale=.75)
    #f,ax = plt.subplots(figsize = (20,9))
    #sns.heatmap(corrmat,vmax=0.8,square=True,cmap = 'PiYG')
    hm = sns.heatmap(cm,cbar =True,cmap = 'PiYG',annot = True, fmt = '.2f',annot_kws = {'size':10},square=True,yticklabels=cols.values,xticklabels=cols.values)

#HeatMap()
#CheckRelationToPrice_Scatter(['TotalBsmtSF'])
#plt.show()