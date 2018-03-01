
# coding: utf-8

# In[18]:


#user input weekday hour and temp, rain
#get the aggre_output from xgboost
from collections import OrderedDict
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn.cross_validation as cv

#from sklearn.cross_validation import cross_val_score

def prediction(weekday,hour,Temp,rain,zipcode): 
    rush = False
    if hour >= 7 & hour <= 10:
        rush = True
    else:
        rush = False
    d = OrderedDict()
    d['Weekday'] = [weekday]
    d['Hour'] = [hour]
    d["Temp"] =[Temp]
    d['Rain'] = [rain]
    d["rush"] = [rush]
    d["zipcode"]= [zipcode]
    
    #make the training dataset
    
    x_new =  pd.DataFrame(data = d)
    nycmodel=pd.read_csv('nycmodeldata.csv',sep='\t', index_col=False, dtype={'zipcode':'S10'})
    add_dummies = pd.get_dummies(nycmodel['zipcode'])
    add_dummies= add_dummies.applymap(np.int)
    nycmodel = pd.concat([nycmodel, add_dummies], axis=1)
    nycmodel.drop(['zipcode','Unnamed: 0'], inplace=True, axis=1)
    length = len(nycmodel.columns[6:]) #98
    zip_str = str(float(zipcode))
    dat = pd.DataFrame(data = np.zeros((1,length),dtype =int), columns = nycmodel.columns[6:])
    dat_col= np.array(dat.columns)
    for i in range(len(dat_col)):
        dat_col[i]= dat_col[i].decode("utf-8") 
    index_0 = np.where(dat_col == zip_str)[0][0]
    dat.iloc[:,index_0] = int(1)
    
    dat2 = pd.concat([x_new,dat],axis = 1)
    dat2.drop(['zipcode'],inplace=True, axis=1)
    
    
    #fit the model
    target=nycmodel[['count']]
    data=nycmodel[[col for col in nycmodel.columns if col not in ['count']]]
    x_train, x_test, y_train, y_test = cv.train_test_split(data, target, test_size=2.0/10, random_state=0)
    XGB=xgb.XGBRegressor(max_depth=14,learning_rate=0.1186,n_estimators=463,silent=True,
            nthread=-1,gamma=1.0,min_child_weight=6.1929,subsample=0.9675,colsample_bytree=0.8544)
    XGB.fit(x_train, y_train)

    pred_y_xgb = XGB.predict(dat2)
    return((zipcode, pred_y_xgb[0]))  
    #give specific predictions on a location, just a number
     
    #give suggestions on other possible pickup locations at the same weekday, hour
    #(a list of zipcodes with the number of pickups )
#filter the predicted table???? or  with the same weekday, hour

#return a list or a map with number of pickups


# In[19]:


prediction(5,8,60,0.2,10002)


# In[20]:


def prediction_nozip(weekday,hour):
    #dat = output_aggre.loc[(output_aggre['Weekday'] == weekday )&( output_aggre['Hour'] == hour)]
    nycmodel= pd.read_csv('nycmodeldata.csv', sep='\t', index_col=False, dtype={'zipcode':'S10'})
    dat = nycmodel.loc[(nycmodel['Weekday'] == weekday) & (nycmodel['Hour'] == hour)]
    dat.drop(['Unnamed: 0'], inplace=True, axis=1)
    return(dat)


# In[21]:


prediction_nozip(5,23)

