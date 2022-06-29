import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# import os
# os.getcwd
# os.chdir('C:\\Users\\ABHI ALEXY\\Desktop\\vs code\\placementAnalysis')

Data=pd.read_csv("placementdata_EEE.csv")
# print(Data.head())
Data_copy=Data.copy()
Data_copy.fillna(value=0,inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Data_copy['Placed']=le.fit_transform(Data_copy['Placed'])
Data_copy=Data_copy.astype({'No_of_projects' : 'int'})
Data_copy=Data_copy.astype({'No_of_Interships' : 'int'})
Data_copy.dtypes
# print(Data_copy.head())
Data_copy.drop(['S No','Roll_Number	','Name'], axis=1,inplace=True,errors='ignore')
Data_copy.pop('Roll_Number')
# print(Data_copy.corr())
Q1=Data_copy.No_of_Interships.quantile(0.25)
Q3=Data_copy.No_of_Interships.quantile(0.75)

IQR=Q3-Q1

lower_limit  =  Q1-1.5*IQR
upper_limit = Q3+1.5*IQR

filter = (Data_copy.No_of_Interships >lower_limit) & (Data_copy.No_of_Interships<upper_limit) 
Data_filtered=Data_copy.loc[filter]
Data_filtered.head()

# plt.boxplot(Data_filtered['No_of_Interships'])

X=Data_filtered.drop(['Placed'],axis=1)
y=Data_filtered.Placed

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
rt = RandomForestClassifier(n_estimators = 100)

rt.fit(X_train , y_train)
y_pred = rt.predict(X_test)

print("Accuracy", metrics.accuracy_score(y_test , y_pred)*100)

# X=[[8,4,4,4]]
# pro=rt.predict_proba(X)[:,1]
# print(rt.predict(X),pro*100)

import pickle
pickle.dump(rt, open('RandomForestClassifiermodel.pkl','wb'))

# Loading model to compare the results
rtmodel = pickle.load(open('RandomForestClassifiermodel.pkl','rb'))
