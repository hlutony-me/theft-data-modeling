# -*- coding: utf-8 -*-
"""
@author: 
"""

# -*- coding: utf-8 -*-
"""
@author: group5
"""

"""1 Data exploration"""
"""1.1 Load Data"""
import pandas as pd
import os
path = "F:/study/2021 fall/comp309-data-warehouse/assignment/group assignment"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
data_group5_theft = pd.read_csv(fullpath,dtype={'city': "string"})
data_group5_theft.columns.values



"""General idea of the dataframe"""
data_group5_theft.columns.values
data_group5_theft.shape
data_group5_theft.describe()
data_group5_theft.dtypes
data_group5_theft.head(5)
data_group5_theft.isnull().sum()
print(data_group5_theft['Status'].value_counts()) #Count on Target
#Drop Status = unknown
data_group5_theft = data_group5_theft.drop(data_group5_theft[data_group5_theft.Status == "UNKNOWN"].index)

#Drop IDs
data_group5_theft=data_group5_theft.drop('ObjectId2', 1)
data_group5_theft=data_group5_theft.drop('OBJECTID', 1)
data_group5_theft=data_group5_theft.drop('event_unique_id', 1)

pd.set_option('display.max_columns',100) #Check the average of all the numeric columns
print(data_group5_theft.groupby('Status').mean())


"""1.2 Statistical assessments including means, averages, and correlations."""
#1.2.1 Occurrence_DayOfWeek
data_week_status = data_group5_theft[["Occurrence_DayOfWeek","Status"]]

data_group5_theft.groupby(['Occurrence_DayOfWeek', 'Status']).size().unstack() #Count occurrence
data_group5_theft.groupby(['Occurrence_DayOfWeek'])['Status'].value_counts(normalize=True) * 100

data_month_status = data_group5_theft[["Occurrence_DayOfWeek","Status"]]
data_month_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

#draw a stacked bar chart 
import matplotlib.pyplot as plt
pd.crosstab(data_group5_theft["Occurrence_DayOfWeek"],data_group5_theft["Status"]).plot(kind='bar')

table=pd.crosstab(data_group5_theft["Occurrence_DayOfWeek"],data_group5_theft["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in weeks')
plt.xlabel('Occurrence_DayOfWeek')
plt.ylabel('Frequency of Stolen')

#1.2.2 Occurrence_Month
data_group5_theft.groupby(['Occurrence_Month', 'Status']).size().unstack() #Count occurrence

data_month_status = data_group5_theft[["Occurrence_Month","Status"]]
data_month_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

#draw a stacked bar chart 
import matplotlib.pyplot as plt
pd.crosstab(data_group5_theft["Occurrence_Month"],data_group5_theft["Status"]).plot(kind='bar')

table=pd.crosstab(data_group5_theft["Occurrence_Month"],data_group5_theft["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in months,')
plt.xlabel('Occurrence_Month')
plt.ylabel('Frequency of Stolen')


#1.2.3 Hood_ID & NeighbourhoodName
data_hood_id_name = data_group5_theft[["Hood_ID","NeighbourhoodName"]] #Decide which to use
data_hood_id_name.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

data_group5_theft.groupby(['NeighbourhoodName', 'Status']).size().unstack() #Count occurrence

data_month_status = data_group5_theft[["NeighbourhoodName","Status"]]
data_month_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

#draw a stacked bar chart 
import matplotlib.pyplot as plt
pd.crosstab(data_group5_theft["NeighbourhoodName"],data_group5_theft["Status"]).plot(kind='bar')

table=pd.crosstab(data_group5_theft["NeighbourhoodName"],data_group5_theft["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in neighbourhood,')
plt.xlabel('NeighbourhoodName')
plt.ylabel('Frequency of Stolen')

# 1.2.4 Location_Type  @Munther
data_group5_theft.groupby(['Location_Type', 'Status']).size().unstack() #Count occurrence

data_Location_Type_status = data_group5_theft[["Location_Type","Status"]]
data_Location_Type_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

#Plot a histogram showing Status by Location_Type category
import matplotlib.pyplot as plt
pd.crosstab(data_Location_Type_status["Location_Type"],data_Location_Type_status["Status"]).plot(kind='bar')

#draw a stacked bar chart of the marital status and the Location_Type of term bicycle thefts to see whether 
#this can be a good predictor of the outcome
table=pd.crosstab(data_Location_Type_status["Location_Type"],data_Location_Type_status["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in Location_Type,')
plt.xlabel('Location_Type')
plt.ylabel('Frequency of Stolen')


# 1.2.5 Premises_Type  @Munther
data_group5_theft.groupby(['Premises_Type', 'Status']).size().unstack() #Count occurrence

data_Premises_Type_status = data_group5_theft[["Premises_Type","Status"]]
data_Premises_Type_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

#draw a stacked bar chart of the marital status and the Premises_Type of term bicycle thefts to see whether 
#this can be a good predictor of the outcome 
pd.crosstab(data_Premises_Type_status["Premises_Type"],data_Premises_Type_status["Status"]).plot(kind='bar')

table=pd.crosstab(data_Premises_Type_status["Premises_Type"],data_Premises_Type_status["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in Premises_Type,')
plt.xlabel('Location_Type')
plt.ylabel('Frequency of Premises_Type')


#Select from premises type and location type
data_premises_location = data_group5_theft[["Premises_Type","Location_Type"]] #Decide which to use
data_premises_location.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation


# 1.2.6 Bike_Make  @Han

# Replace missing value
data_group5_theft["Bike_Make"].isna().sum()
data_make = data_group5_theft[["Bike_Make"]]

selected_rows_make = data_make[data_make['Bike_Make'].notnull()]

from numpy import random as rd
replacement =rd.choice(selected_rows_make["Bike_Make"], size=data_group5_theft["Bike_Make"].isna().sum(), replace=False).tolist()
print(replacement)

data_group5_theft.loc[data_group5_theft['Bike_Make'].isna(), 'Bike_Make'] = replacement

data_make_status = data_group5_theft[["Bike_Make","Status"]]
data_make_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

data_group5_theft.groupby(["Bike_Make","Status"]).size().unstack() #Count occurrence


#draw a stacked bar chart 
import matplotlib.pyplot as plt
pd.crosstab(data_group5_theft["Bike_Make"],data_group5_theft["Status"]).plot(kind='bar')

table=pd.crosstab(data_group5_theft["Bike_Make"],data_group5_theft["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in makes,')
plt.xlabel('Bike_Make')
plt.ylabel('Frequency of Stolen')


#draw plot on top acured makes
top_make = []
for key, value in data_make['Bike_Make'].value_counts().head(10).iteritems():
    top_make.append(key)

print(top_make)

data_make_status_top=data_make_status[data_make_status['Bike_Make'].isin(top_make)]

import matplotlib.pyplot as plt
pd.crosstab(data_make_status_top["Bike_Make"],data_make_status_top["Status"]).plot(kind='bar')

table=pd.crosstab(data_make_status_top["Bike_Make"],data_make_status_top["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in top makes,')
plt.xlabel('Bike_Make')
plt.ylabel('Frequency of Stolen')

# 1.2.6 Bike_Model  @Han

data_group5_theft["Bike_Model"].isna().sum()
data_group5_theft['Bike_Model'] = data_group5_theft['Bike_Model'].fillna("UNKNOWN")

data_model_status=  data_group5_theft[["Bike_Model","Status"]]
data_model_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

data_group5_theft.groupby(["Bike_Model","Status"]).size().unstack() #Count occurrence

#draw a stacked bar chart 
import matplotlib.pyplot as plt
pd.crosstab(data_group5_theft["Bike_Model"],data_group5_theft["Status"]).plot(kind='bar')

table=pd.crosstab(data_group5_theft["Bike_Model"],data_group5_theft["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in bike model,')
plt.xlabel('Bike_Model')
plt.ylabel('Frequency of Stolen')

#draw plot on top acured models
top_model = []
for key, value in data_model_status['Bike_Model'].value_counts().head(10).iteritems():
    top_model.append(key)

print(top_model)

data_model_status_top=data_model_status[data_model_status['Bike_Model'].isin(top_model)]

import matplotlib.pyplot as plt
pd.crosstab(data_model_status_top["Bike_Model"],data_model_status_top["Status"]).plot(kind='bar')

table=pd.crosstab(data_model_status_top["Bike_Model"],data_model_status_top["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in top models,')
plt.xlabel('Bike_Model')
plt.ylabel('Frequency of Stolen')




#1.2.7 Bike_Type @Amel
data_month_status = data_group5_theft[["Bike_Type","Status"]] #Decide which to use
data_month_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

data_group5_theft.groupby(['Bike_Type', 'Status']).size().unstack() #Count occurrence

data_month_status = data_group5_theft[["Bike_Type","Status"]]
data_month_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

#draw a stacked bar chart 
import matplotlib.pyplot as plt
pd.crosstab(data_group5_theft["Bike_Type"],data_group5_theft["Status"]).plot(kind='bar')

table=pd.crosstab(data_group5_theft["Bike_Type"],data_group5_theft["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in neighbourhood,')
plt.xlabel('Bike_Type')
plt.ylabel('Frequency of Stolen')

#1.2.8 Bike_Speed @Amel
data_speed_status = data_group5_theft[["Bike_Speed","Status"]] #Decide which to use
data_speed_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

data_group5_theft.groupby(['Bike_Speed', 'Status']).size().unstack() #Count occurrence

import pandas as pd
import numpy as np
data_group5_theft["Bike_Speed"].max()
data_group5_theft["Bike_Speed"].min()
data_group5_theft["Bike_Speed"].mean()
bins= [0,10,20,30,40,50,60,70,80,90,100]
data_speed_status['Speed_Range'] = pd.cut(data_group5_theft["Bike_Speed"], bins=bins,include_lowest=True)

data_speed_status.groupby(['Speed_Range', 'Status']).size().unstack() #Count occurrence
data_speed_status.groupby(['Speed_Range'])['Status'].value_counts(normalize=True) * 100



data_speed_status = data_group5_theft[["Bike_Speed","Status"]]
data_speed_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

#draw a stacked bar chart 
import matplotlib.pyplot as plt
#chart on speed range
pd.crosstab(data_speed_status["Speed_Range"],data_speed_status["Status"]).plot(kind='bar')

#chart on speed
pd.crosstab(data_group5_theft["Bike_Speed"],data_group5_theft["Status"]).plot(kind='bar')

#chart on speed range
table=pd.crosstab(data_speed_status["Speed_Range"],data_speed_status["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in speed,')
plt.xlabel('Bike_Speed_Range')
plt.ylabel('Frequency of Stolen')

#chart on speed
table=pd.crosstab(data_group5_theft["Bike_Speed"],data_group5_theft["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in speed,')
plt.xlabel('Bike_Speed')
plt.ylabel('Frequency of Stolen')

#1.2.9 Bike_Colour @Ilah
#Missing values

data_group5_theft["Bike_Colour"].isna().sum()
data_group5_theft['Bike_Colour'] = data_group5_theft['Bike_Colour'].fillna("UNKNOWN")


data_colour_status = data_group5_theft[["Bike_Colour","Status"]]

data_group5_theft.groupby(['Bike_Colour', 'Status']).size().unstack() #Count occurrence
data_group5_theft.groupby(['Bike_Colour'])['Status'].value_counts(normalize=True) * 100

data_colour_status = data_group5_theft[["Bike_Colour","Status"]]
data_colour_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation

#draw a stacked bar chart for Bike_Colour
import matplotlib.pyplot as plt
pd.crosstab(data_group5_theft["Bike_Colour"],data_group5_theft["Status"]).plot(kind='bar')

table=pd.crosstab(data_group5_theft["Bike_Colour"],data_group5_theft["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in neighbourhood,')
plt.xlabel('Bike_Colour')
plt.ylabel('Frequency of Stolen')




#1.2.10 Cost_of_Bike  @Ilah 

# Missing data evaluations
data_group5_theft["Cost_of_Bike"].isnull().sum()
data_make = data_group5_theft[["Cost_of_Bike"]]

selected_rows_make = data_make[data_make['Cost_of_Bike'].notnull()]
print (selected_rows_make)

from numpy import random as rd
replacement =rd.choice(selected_rows_make["Cost_of_Bike"], size=data_group5_theft["Cost_of_Bike"].isna().sum(), replace=False).tolist()
print(replacement)
 
data_group5_theft.loc[data_group5_theft['Cost_of_Bike'].isna(), 'Cost_of_Bike'] = replacement



data_cost_status = data_group5_theft[["Cost_of_Bike","Status"]]
data_cost_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) # Calculate correlation


import pandas as pd

data_group5_theft["Cost_of_Bike"].max()
data_group5_theft["Cost_of_Bike"].min()
data_group5_theft["Cost_of_Bike"].mean()
bins= [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,np.inf]
data_cost_status['Cost_Range'] = pd.cut(data_group5_theft["Cost_of_Bike"], bins=bins,include_lowest=True)

data_cost_status.groupby(['Cost_Range', 'Status']).size().unstack() #Count occurrence
data_cost_status.groupby(['Cost_Range'])['Status'].value_counts(normalize=True) * 100

#draw a stacked bar chart for Cost_of_Bike
pd.crosstab(data_cost_status["Cost_Range"],data_cost_status["Status"]).plot(kind='bar')


table=pd.crosstab(data_cost_status["Cost_Range"],data_cost_status["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in cost')
plt.xlabel('Cost_of_Bike')
plt.ylabel('Frequency of Stolen')



#1.2.11 Longitude @David
data_group5_theft[["Longitude","Status"]].isna().sum()

data_Longitude_status = data_group5_theft[["Longitude","Status"]]
data_Longitude_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) 

data_Longitude_status[["Longitude"]].min()
data_Longitude_status[["Longitude"]].max()
import pandas as pd
import numpy as np

bins= [-79.73,-79.53,-79.33,-79.13,-78.93,-78.73]
data_Longitude_status['Longitude_Range'] = pd.cut(data_Longitude_status["Longitude"], bins=bins,include_lowest=True)

data_Longitude_status.groupby(['Longitude_Range', 'Status']).size().unstack() #Count occurrence
data_Longitude_status.groupby(['Longitude_Range'])['Status'].value_counts(normalize=True) * 100


#draw a stacked bar chart 
import matplotlib.pyplot as plt
#chart on speed range
pd.crosstab(data_Longitude_status["Longitude_Range"],data_Longitude_status["Status"]).plot(kind='bar')


"""
import matplotlib.pyplot as plt
pd.crosstab(data_Longitude_status["Longitude"],data_Longitude_status["Status"]).plot(kind='bar')
table=pd.crosstab(data_Longitude_status["Longitude"],data_Longitude_status["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in Longitude,')
plt.xlabel('Longitude')
plt.ylabel('Frequency of Stolen')
"""





#1.2.12 Latitude @David
data_Latitude_status = data_group5_theft[["Latitude","Status"]]
data_Latitude_status.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1) 

data_Latitude_status[["Latitude"]].min()
data_Latitude_status[["Latitude"]].max()
import pandas as pd
import numpy as np

bins= [43.3,43.5,43.7,43.9,44.1]
data_Latitude_status['Latitude_Range'] = pd.cut(data_Latitude_status["Latitude"], bins=bins,include_lowest=True)

data_Latitude_status.groupby(['Latitude_Range', 'Status']).size().unstack() #Count occurrence
data_Latitude_status.groupby(['Latitude_Range'])['Status'].value_counts(normalize=True) * 100


#draw a stacked bar chart 
import matplotlib.pyplot as plt
#chart on speed range
pd.crosstab(data_Latitude_status["Latitude_Range"],data_Latitude_status["Status"]).plot(kind='bar')



"""
import matplotlib.pyplot as plt
pd.crosstab(data_Latitude_status["Latitude"],data_Latitude_status["Status"]).plot(kind='bar')
table=pd.crosstab(data_Latitude_status["Latitude"],data_Latitude_status["Status"])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stolen rate in Latitude,')
plt.xlabel('Latitude')
plt.ylabel('Frequency of Stolen')

"""

#Laitude & Longitude
data_Longitude_latitude_status = data_group5_theft[["Latitude","Longitude","Status"]]
data_group5_theft["Longitude"].max()
data_group5_theft["Longitude"].min()

import numpy as np

data_Longitude_latitude_status["Status"].value_counts()

color=[]
for var in data_group5_theft["Status"].values:
    if var =='UNKNOWN':
        color.append('r')
    if var =='STOLEN':
        color.append('g')
    if var =='RECOVERED':
        color.append('b')
data_Longitude_latitude_status['color']=color        
        
data_group5_theft.plot(kind="scatter", x="Longitude", y="Latitude",c=color,alpha=0.4)

#
#



"""2 Data Cleaning"""


#Encode Features

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
data_group5_theft_target_encoded=data_group5_theft

data_group5_cat = data_group5_theft.select_dtypes(include=['object'])
data_group5_num = data_group5_theft.select_dtypes(exclude=['object'])


for var in data_group5_cat.columns.values.tolist():
    data_group5_cat[var] =  ord_enc.fit_transform(data_group5_cat[[var]].astype(str))

data_group5_theft_encoded=data_group5_cat.join(data_group5_num)



#Balance classes

# Separate predictors and target
x=data_group5_theft_encoded.drop('Status', 1)
y=data_group5_theft_encoded.Status
# handle inbalanced dataset for Status (STOLEN is way more than RECOVERED)
from imblearn.over_sampling import SMOTE 
from collections import Counter
X= data_group5_theft.drop('Status', 1)
Y=data_group5_theft.Status
data_group5_theft.dtypes

oversample = SMOTE() #SMOTENC(random_state=42, categorical_features=[2,3,5,6,10,12,13,17,18,19,20,21,22,23,24,25,27,29])
x_ros, y_ros = oversample.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))

data_group5_theft_encoded = x_ros.join(y_ros)


# example of a normalization

from sklearn.preprocessing import MinMaxScaler
# define min max scaler
scaler = MinMaxScaler()
# transform features
import numpy as np

for index,var in enumerate(data_group5_theft_encoded.columns.values.tolist()):
    scaled = scaler.fit_transform(data_group5_theft_encoded[var][:, np.newaxis])
    data_group5_theft_encoded[var] = np.transpose(scaled)[0]







#feature importance
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x_ros,y_ros)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()

# Heatmap on featues 
import seaborn as sns
corrmat = data_group5_theft_encoded.corr()

top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data_group5_theft_encoded[top_corr_features].corr(),annot=True,cmap="RdYlGn")


#sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X=data_group5_theft_encoded.drop('Status', 1)
Y=data_group5_theft_encoded.Status

rfe = RFE(model)
rfe = rfe.fit(X,Y )
print(rfe.support_)
print(rfe.ranking_)

for index,var in enumerate(X.columns.values):
    print(var+'   '+ str(rfe.support_[index])+'   '+str(rfe.ranking_[index]))
    #print (feature + ' ' + rfe.support_[index]+' '+rfe.ranking_[index])
    
#Select predictors and rank features again

X_selected=data_group5_theft_encoded[["Primary_Offence","Occurrence_Month","Occurrence_DayOfWeek","Report_Date","Report_Month","Report_DayOfWeek","NeighbourhoodName","Premises_Type","Bike_Make","Bike_Type","Bike_Colour","Occurrence_DayOfMonth","Occurrence_Hour","Report_Year"]]


rfe = RFE(model)
rfe = rfe.fit(X_selected,Y )
print(rfe.support_)
print(rfe.ranking_)

for index,var in enumerate(X_selected.columns.values):
    print(var+'   '+ str(rfe.support_[index])+'   '+str(rfe.ranking_[index]))
    #print (feature + ' ' + rfe.support_[index]+' '+rfe.ranking_[index])
    
X_final = data_group5_theft_encoded[["Primary_Offence","Occurrence_Month","Report_DayOfWeek","Premises_Type","Bike_Type","Bike_Colour"]]


"""Data modeling"""
# Re-encode features using dummies
df_feature_selected=X_final

df_feature_selected = pd.get_dummies(df_feature_selected) 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_feature_selected) 
df_feature_selected=pd.DataFrame(scaled_df, columns=df_feature_selected.columns.values.tolist())



#1- split the data into 80%training and 20% for testing, note  added the solver to avoid warnings
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y , test_size=0.2, random_state=0)
# 2-Let us build the model and validate the parameters
from sklearn import linear_model
from sklearn import metrics
clf1 = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000)
clf1.fit(X_train, Y_train)
#3- Run the test data against the new model
probs = clf1.predict_proba(X_test)
print(probs)
predicted = clf1.predict(X_test)
print (predicted)
#4-Check model accuracy
print (metrics.accuracy_score(Y_test, predicted))	
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, predicted))



import joblib 
joblib.dump(clf1, 'C:/Users/lhton/Downloads/model_group_5.pkl')
print("Model dumped!")


model_columns = list(X_final.columns)
print(model_columns)
joblib.dump(model_columns, 'C:/Users/lhton/Downloads/columns_group_5.pkl')
print("Models columns dumped!")





"""
"""2 Data Modeling"""
#2.1 Data transformations
cat_feature_vars=['Primary_Offence','Report_Hour','Occurrence_Hour','Report_DayOfMonth','NeighbourhoodName','Premises_Type','Bike_Model','Bike_Make','Bike_Type','Bike_Colour']
cat_target_var=["Status"]

#Encode Target
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
data_group5_theft_target_encoded=data_group5_theft[cat_target_var]
data_group5_theft_target_encoded["Status_Code"] = ord_enc.fit_transform(data_group5_theft_target_encoded[["Status"]])



data_group5_theft_target_encoded=data_group5_theft[cat_target_var]
data_group5_theft_target_encoded["Status_Code"] = pd.get_dummies(data_group5_theft_target_encoded[["Status"]])

#Encode Categorical Features
data_group5_theft_encoded_feature = data_group5_theft[cat_feature_vars]

for var in cat_feature_vars:
  data_group5_theft_encoded_feature[var+"_Code"] = ord_enc.fit_transform(data_group5_theft_encoded_feature[[var]])

to_keep=[i for i in data_group5_theft_encoded_feature.columns.values.tolist() if i not in cat_feature_vars]
data_group5_theft_encoded_feature_final=data_group5_theft_encoded_feature[to_keep]



#Append numerical columns to the data set
data_group5_theft_encoded_feature_final=data_group5_theft_encoded_feature_final.join(data_group5_theft['Latitude'])
data_group5_theft_encoded_feature_final=data_group5_theft_encoded_feature_final.join(data_group5_theft['Longitude'])
data_group5_theft_encoded_feature_final=data_group5_theft_encoded_feature_final.join(data_group5_theft['Bike_Speed'])
data_group5_theft_encoded_feature_final=data_group5_theft_encoded_feature_final.join(data_group5_theft['Cost_of_Bike'])


data_group5_theft_encoded_feature_final.head(11)
data_group5_theft_encoded_feature_final_vars=data_group5_theft_encoded_feature_final.columns.values.tolist()
 


# example of a normalization

from sklearn.preprocessing import MinMaxScaler
# define min max scaler
scaler = MinMaxScaler()
# transform features
import numpy as np

for index,var in enumerate(data_group5_theft_encoded_feature_final_vars):
    scaled = scaler.fit_transform(data_group5_theft_encoded_feature_final[var][:, np.newaxis])
    data_group5_theft_encoded_feature_final[var+'_Normalized'] = np.transpose(scaled)[0]

print(data_group5_theft_encoded_feature_final.columns.values.tolist())

to_keep=[i for i in data_group5_theft_encoded_feature_final.columns.values.tolist() if i not in data_group5_theft_encoded_feature_final_vars]
data_group5_theft_encoded_normalized_feature_final=data_group5_theft_encoded_feature_final[to_keep]

data_group5_theft_encoded_normalized_feature_final.columns.values.tolist()



#Select features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


X=data_group5_theft_encoded_normalized_feature_final[to_keep]
Y=data_group5_theft_target_encoded['Status_Code']

model = LogisticRegression(solver='lbfgs', max_iter=10000)
rfe = RFE(model)
rfe = rfe.fit(X,Y )
print(rfe.support_)
print(rfe.ranking_)

for index,var in enumerate(data_group5_theft_encoded_feature_final_vars):
    print(var+'   '+ str(rfe.support_[index])+'   '+str(rfe.ranking_[index]))
    #print (feature + ' ' + rfe.support_[index]+' '+rfe.ranking_[index])
    
data_group5_theft_encoded_feature_final_selected = data_group5_theft_encoded_normalized_feature_final[["Premises_Type_Code_Normalized","Bike_Type_Code_Normalized","Bike_Make_Code_Normalized","Latitude_Normalized","Longitude_Normalized","Bike_Speed_Normalized"]]


# Re-encode features using dummies
df_feature_selected=data_group5_theft[["Premises_Type","Bike_Type","Bike_Make","Latitude","Longitude","Bike_Speed"]]

df_feature_selected = pd.get_dummies(df_feature_selected) 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_feature_selected) 
df_feature_selected=pd.DataFrame(scaled_df, columns=df_feature_selected.columns.values.tolist())


#1- split the data into 80%training and 20% for testing, note  added the solver to avoid warnings
from sklearn.model_selection import train_test_split

X=df_feature_selected
Y=data_group5_theft_target_encoded['Status_Code']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2, random_state=0)
# 2-Let us build the model and validate the parameters
from sklearn import linear_model
from sklearn import metrics
clf1 = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000)
clf1.fit(X_train, Y_train)
#3- Run the test data against the new model
probs = clf1.predict_proba(X_test)
print(probs)
predicted = clf1.predict(X_test)
print (predicted)
#4-Check model accuracy
print (metrics.accuracy_score(Y_test, predicted))	
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(Y_test, predicted))



import joblib 
joblib.dump(clf1, 'C:/Users/lhton/Downloads/model_group_5.pkl')
print("Model dumped!")


model_columns = list(X.columns)
print(model_columns)
joblib.dump(model_columns, 'C:/Users/lhton/Downloads/columns_group_5.pkl')
print("Models columns dumped!")




"""# handle inbalanced dataset for Status (STOLEN is way more than RECOVERED)
over = SMOTE()
X, y = over.fit_resample(X, y)
Counter(y)
"""
"""