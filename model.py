import pandas as pd
import numpy as np

#import data
dataset = pd.read_csv('Train_dataset.csv')
dataset1 = pd.read_csv('Test_dataset.csv')

#fill nan values with string ""
reg = dataset.iloc[:,1]
gen = dataset.iloc[:,2]
mar = dataset.iloc[:,5]
occ = dataset.iloc[:,7]
occ = occ.fillna('')
modet = dataset.iloc[:,8]
modet = modet.fillna('')
com = dataset.iloc[:,11]
com = com.fillna('')
pulm = dataset.iloc[:,14]
pulm = pulm.fillna('')
cardio = dataset.iloc[:,15]
cardio = cardio.fillna('')

reg1 = dataset1.iloc[:,1]
gen1 = dataset1.iloc[:,2]
mar1 = dataset1.iloc[:,5]
occ1 = dataset1.iloc[:,7]
occ1 = occ1.fillna('')
modet1 = dataset1.iloc[:,8]
modet1 = modet1.fillna('')
com1 = dataset1.iloc[:,11]
com1 = com1.fillna('')
pulm1 = dataset1.iloc[:,14]
pulm1 = pulm1.fillna('')
cardio1 = dataset1.iloc[:,15]
cardio1 = cardio1.fillna('')

#labal enconding and fit in data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()

dataset['Region'] = labelencoder_Y.fit_transform(reg)
dataset['Gender'] = labelencoder_Y.fit_transform(gen)
dataset['Married'] = labelencoder_Y.fit_transform(mar)
dataset['Occupation'] = labelencoder_Y.fit_transform(occ)
dataset['Mode_transport'] = labelencoder_Y.fit_transform(modet)
dataset['comorbidity'] = labelencoder_Y.fit_transform(com)
dataset['Pulmonary score'] = labelencoder_Y.fit_transform(pulm)
dataset['cardiological pressure'] = labelencoder_Y.fit_transform(cardio)

dataset1['Region'] = labelencoder_Y.fit_transform(reg1)
dataset1['Gender'] = labelencoder_Y.fit_transform(gen1)
dataset1['Married'] = labelencoder_Y.fit_transform(mar1)
dataset1['Occupation'] = labelencoder_Y.fit_transform(occ1)
dataset1['Mode_transport'] = labelencoder_Y.fit_transform(modet1)
dataset1['comorbidity'] = labelencoder_Y.fit_transform(com1)
dataset1['Pulmonary score'] = labelencoder_Y.fit_transform(pulm1)
dataset1['cardiological pressure'] = labelencoder_Y.fit_transform(cardio1)

# drop columns
dataset = dataset.drop(columns=['people_ID','Children','Designation', 'Name'])
dataset1 = dataset1.drop(columns=['people_ID','Children','Designation', 'Name'])

#fill missing values with mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(dataset.iloc[:, 12:23])
dataset.iloc[:, 12:23] = imputer.transform(dataset.iloc[:, 12:23])
imputer = imputer.fit(dataset1.iloc[:, 12:23])
dataset1.iloc[:, 12:23] = imputer.transform(dataset1.iloc[:, 12:23])

#initialize X_train, Y_train, X_test
X_train = dataset.iloc[:, :-1]
Y_train = dataset.iloc[:,23]

X_test = dataset1

#standard scaling of data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#decisionn tree algorithm
from sklearn.tree import DecisionTreeRegressor
clf1=DecisionTreeRegressor()
clf1.fit(X_train,Y_train)

# Predicting the Test set results
y_pred = clf1.predict(X_test)

