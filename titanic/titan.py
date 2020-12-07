import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
titanic = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all_data = pd.concat([titanic,test], ignore_index = True)

plt.rcParams['axes.unicode_minus'] = False

#sns.barplot(x = 'Survived',y = 'Sex', data = all_data)
#sns.barplot(x = 'Embarked', y = 'Survived', data = all_data)
#sur_age = sns.FacetGrid(all_data, col='Survived')
#sns.countplot('Sex',hue = 'Pclass', data = all_data)
###################### PClass ######################################
#PclassDf 




####################Ticket ############################################
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])
def ticket_label(s):
    if s == 1: return 1
    elif 2 <= s <= 4: return 2
    elif 4 < s <= 8: return 3
    else: return 4
all_data['TicketGroup'] = all_data['TicketGroup'].apply(ticket_label)
sns.barplot(x = 'TicketGroup', y = 'Survived', data= all_data)
###################对title进行分类######################################

all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))


TitleDf = pd.DataFrame()
TitleDf['Title'] = all_data['Title'].map(Title_Dict)
TitleDf = pd.get_dummies(TitleDf.Title)
print(TitleDf.head())
all_data = pd.concat([all_data,TitleDf],axis= 1)
#all_data.drop('Title',axis= 1,inplace=True)
#sns.barplot(x="Title", y="Survived", data=all_data)
all_data.drop('Name',axis= 1, inplace=True)
#plt.show()

###################### Family Size ##############################

all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

def fam_label(s):
    if s== 1: return 1
    elif 2 <= s < 4: return 2
    elif 4 <= s <7 : return 3
    else: return 4

all_data['FamilyLabel'] = all_data['FamilySize'].apply(fam_label)
familyDf = pd.get_dummies(all_data['FamilyLabel'])
sns.barplot(x = 'FamilyLabel', y = 'Survived', data=all_data)


###################通过 Pclass Sex 补充age##############################
age_df = all_data[['Age','Pclass','Sex','Title']]
age_df = pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y = known_age[:,0]
x = known_age[:,1:]
rfr = RandomForestRegressor(random_state=0,n_estimators=100,n_jobs=-1)
rfr.fit(x,y)
predictAges = rfr.predict(unknown_age[:,1:])
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictAges
#print(Ticket_Count)
#print(all_data.groupby(by = ['Pclass','Embarked']).Fare.median())
#sur_age.map(plt.hist,'Age',bins = 15)



################## one-hot code for Sex#########################
sex_map = {'male':1,'female':0}
all_data.Sex = all_data['Sex'].map(sex_map)
#SexDf = pd.get_dummies(all_data.Sex,prefix='Sex')
#all_data = pd.concat([all_data,SexDf],axis= 1)
#all_data.drop('Sex',axis = 1, inplace= True)



################ one-hot code for Embarked and Pclass #####################
all_data['Embarked'] = all_data['Embarked'].fillna('C')
all_data['Fare'] = all_data['Fare'].fillna('8')
#plt.show()
EmbarkedDf = pd.get_dummies(all_data['Embarked'], prefix = 'Embarked')
PclassDf = pd.get_dummies(all_data['Pclass'], prefix='Pclass')
all_data = pd.concat([all_data,PclassDf],axis=1)
all_data = pd.concat([all_data,EmbarkedDf],axis= 1)
all_data.drop('Embarked',axis = 1,inplace=True)
all_data.drop('Pclass',axis = 1,inplace=True)


################Cabin fill #####################
all_data.Cabin = all_data.Cabin.fillna('U')
all_data.Cabin = all_data.Cabin.map(lambda x:x[0])
CabinDf = pd.get_dummies(all_data.Cabin, prefix='Cabin')

all_data = pd.concat([all_data,CabinDf], axis= 1)
all_data.drop('Cabin',axis= 1, inplace=True)

#print(EmbarkedDf.head())
#plt.show()
corrDf = all_data.corr()
#print(corrDf)
#print(all_data.head(10))
all_data.info()
print(all_data.describe())

######################## generate new csv#####################

handled = pd.concat([TitleDf,PclassDf,familyDf,all_data['Fare'],CabinDf,EmbarkedDf,all_data['Sex']],axis=1)
#print(handled.head(10))
SourceRow = 891
source_X = handled.loc[0:SourceRow-1, :]
#print('Source_X: \n\n',source_X)
source_Y = all_data.loc[0:SourceRow-1,'Survived']
#print('Source_Y \n\n',source_Y)
pred_X = handled.loc[SourceRow:,:]
#pred_X.to_csv('predx.csv',index = False)
#print('Predict_X \n\n',pred_X)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
trainX,testX,trainY,testY = train_test_split(source_X,source_Y,train_size =0.8)
model.fit(trainX,trainY)

print(model.score(testX,testY))
#pred_X = pred_X.dropna()
pred_Y = model.predict(pred_X)
pred_Y = pred_Y.astype(int)
passenger_id = all_data.loc[SourceRow:,'PassengerId']
predDf = pd.DataFrame({'PassengerId':passenger_id, 'Survived':pred_Y})

print(predDf.head(20))

predDf.to_csv('result.csv',index = False)

