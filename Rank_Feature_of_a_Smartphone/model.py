# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,classification_report
import pickle

# Loading data
df_train = pd.read_csv('train.csv')


df_train["rank_by_price"] = df_train["price_range"].rank()
df1 = df_train
df1

#Sorting above dataset according to ranked_price_range

df1.sort_values(by=["rank_by_price"])

#Ranking on all the features using rank()

RankedData = df1.rank()
RankedData.sort_values(by="price_range")

#Ranking all the features separately to correct output
#Because not all features are good when values are high or low
#It depends on each and every feature.

r = df1
r["rank_by_price"] = r["price_range"].rank()
r["rank_by_battery"] = r["battery_power"].rank(ascending=False)
r["rank_by_blueooth"] = r["blue"].rank(ascending=False)
r["rank_by_clockspeed"] = r["clock_speed"].rank(ascending=False)
r["rank_by_DualSIM"] = r["dual_sim"].rank(ascending=False)
r["rank_by_fc"] = r["fc"].rank(ascending=False)
r["rank_by_4G"] = r["four_g"].rank(ascending=False)
r["rank_by_InternalMemory"] = r["int_memory"].rank(ascending=False)
r["rank_by_mdep"] = r["m_dep"].rank(ascending=False)
r["rank_by_weight"] = r["mobile_wt"].rank(ascending=True)
r["rank_by_ncores"] = r["n_cores"].rank(ascending=False)
r["rank_by_pc"] = r["pc"].rank(ascending=False)
r["rank_by_height"] = r["px_height"].rank(ascending=False)
r["rank_by_width"] = r["px_width"].rank(ascending=False)
r["rank_by_ram"] = r["ram"].rank(ascending=False)
r["rank_by_sch"] = r["sc_h"].rank(ascending=False)
r["rank_by_scw"] = r["sc_w"].rank(ascending=False)
r["rank_by_talktime"] = r["talk_time"].rank(ascending=False)
r["rank_by_3G"] = r["three_g"].rank(ascending=False)
r["rank_by_touchscreen"] = r["touch_screen"].rank(ascending=False)
r["rank_by_wifi"] = r["wifi"].rank(ascending=False)
r.head()

data = pd.read_csv('train.csv')

# Split the dataset into features and target
x = data.drop('price_range', axis=1)
y = data['price_range']

xtrain,xtest,ytrain,ytest =train_test_split(x,y, test_size=0.25,random_state = 42)

# Min-max scaling

minmax = MinMaxScaler(feature_range=(0,1))  #creating instance
minmax.fit(xtrain)
xtrain = minmax.transform(xtrain)
xtest = minmax.transform(xtest) 


# Classification Model
# Logistic Regression

lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)
ypred_lr = model.predict(xtest)
print('Accuracy score is:',accuracy_score(ytest,ypred_lr))

#feature importance and ranking
coefficients = model.coef_

avg_importance =np.mean(np.abs(coefficients),axis=0)
feature_importance = pd.DataFrame({'Feature': x.columns, 'Importance': avg_importance})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance .sort_values(by=['Importance'],ascending=False,inplace=True)
feature_importance['rank']= feature_importance['Importance'].rank(ascending=False)
feature_importance


#Pickling is a way to convert a python object (list, dict, etc.) into a character stream
# Stores object data to model.pkl file

pickle.dump(model,open('model.pkl','wb'))

#for_min-max scaler
pickle.dump(minmax,open('scaling_features.pkl','wb'))