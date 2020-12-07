import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
titanic = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all_data = pd.concat([titanic,test], ignore_index = True)
print(all_data.describe())
def Sex_Survived():
    sns.barplot(x = 'Sex',y='Survived',data=all_data)

def Pclass_Survived():
    sns.barplot(x = 'Pclass',y = 'Survived',data = all_data)

def Age_Survived():
    sur_age = sns.FacetGrid(all_data, col = 'Survived')
    sur_age.map(plt.hist, 'Age',bins = 20)
    sur_age.map(sns.distplot, 'Age', bins = 20)

def Age_Density_Survived():
    sur_age = sns.FacetGrid(all_data,hue = 'Survived', aspect= 2)
    sur_age.map(sns.kdeplot,'Age',shade=True)
    sur_age.set(xlim = (0,all_data['Age'].max()))
    sur_age.add_legend()
    plt.xlabel('Age')
    plt.ylabel('Density')

def Embarked_Survived():
    sns.barplot(x = 'Embarked',y='Survived',data=all_data)
def Embarked():
    sur_emb1 = all_data['Embarked'][all_data['Survived'] == 1].value_counts()
    
def SibSp_Survived():
    sns.barplot(x = 'SibSp', y='Survived', data=all_data)
def Title_Survived():
    all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
    sns.barplot(x = 'Title', y = 'Survived', data = all_data)
#Sex_Survived()
#Age_Survived()
#Age_Density_Survived()
Title_Survived()
plt.show()