
# coding: utf-8

# ##Required Libraries
# * pandas
# * numpy
# * matplotlib 
# * scikit-learn
# * statsmodels

# In[ ]:

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, LinearRegression
import statsmodels.formula.api as smf
# get_ipython().magic(u'matplotlib inline')


# ###Functions that split datetime into date and time and creates submission file

# In[ ]:

#remove warning
# pd.options.mode.chained_assignment = None 

# clearn data and create submission

def nicedata(df):
    
    #split datetime into 2 columns
    time = [item.split(' ')[1] for item in df['datetime']]
    date = pd.Series([item.split(' ')[0] for item in df['datetime']])
    #rename columns
    df['datetime'].update(date)
    df['time'] = time
    df.rename(columns={'datetime':'date'},inplace=True)
    #reorganize columns
    col = df.columns.tolist()
    col = col[-1:]+col[:-1]
    col[0], col[1] = col[1], col[0]
    df = df[col]
    #remove :00:00 on every time
    df['time']=df['time'].str.replace(':00:00','')
    return df

def submitdata(pred,test2,name='submission'):

    #extract name from original 'datetime' feature
    keep = pd.read_csv('data/test.csv')
    keep = keep['datetime']
    
    #save to file
    submit = pd.concat([keep,pred],axis=1)
    submit.columns=['datetime','count']
    submit.to_csv('data/'+name+'.csv',index=False)

    
#this removes warning that will otherwise come about
pd.options.mode.chained_assignment = None


# ##Plot Data
# Here we will plot the data for 4 days.  Days are _NOT_ always 24 hours, sometimes data is missing.

# In[ ]:

data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
data = nicedata(data)
test2 = nicedata(test)
print(test.shape)
#print(test.head(3))
x=0
day_1 = data[0:24]
day_2 = data[24:47]
day_3 = data[47:69]
day_4 = data[69:92]
# print(day_4)


f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
ax1.plot(day_1.time,day_1['count'])
ax1.set_title('2 Weekends, 2 Working days')
ax2.plot(day_2.time,day_2['count'])
ax3.plot(day_3.time,day_3['count'])
ax4.plot(day_4.time,day_4['count'])
f.subplots_adjust(hspace=0)
f.set_size_inches(15,10)

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.

# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)


# As you can see, for working days (the second 2) there is a clear spike during morning and afternoon rushes.

# In[ ]:

# 4 x 4 subplot decided not to use
# plt.rc('lines', linewidth=3.0)
# plt.figure(num=None, figsize=(16, 10), dpi=100, facecolor='w', edgecolor='k')
# # plt.subplot(2,2,1).set_title('day 1')
# # plt.plot(day_1.time,day_1['count'])
# # plt.subplot(2,2,2).set_title('day 2')
# # plt.plot(day_2.time,day_2['count'])
# # plt.subplot(2,2,3).set_title('day 3')
# # plt.plot(day_3.time,day_3['count'])
# # plt.subplot(2,2,4).set_title('day 4')
# # plt.plot(day_4.time,day_4['count'])


# Here is a variety of functions we can choose from, but lets ignore these for now.

# In[5]:

svr_rbf = SVR()
print(data.columns)
train_rbf = data[['time','workingday','temp','humidity']]
target_rbf = data['count']
test_rbf = test2[['time','workingday','temp','humidity']]
# print(target_rbf.head(3))


# In[ ]:

y_rbf = svr_rbf.fit(train_rbf,target_rbf)
pred = y_rbf.predict(test_rbf)


# In[33]:

# print(pred)


# In[41]:

predfinal = pd.Series(pred)
#extract name from original 'datetime' feature
keep = pd.read_csv('data/test.csv')
keep = keep['datetime']

# # #save to file
# submit = pd.concat([keep,predfinal],axis=1)
# submit.columns=['datetime','count']
# submit.to_csv('data/submitrbf.csv',index=False)


# Plot day with SVM vs actual

# In[ ]:

day_3 = data[47:69]
testday_1 = test2[0:24]
# print(day_4)

f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(day_3.time,day_3['count'])
# ax1.set_title('2 Weekends, 2 Working days')
ax2.plot(testday_1.time,predfinal[0:24])
# ax3.plot(day_3.time,day_3['count'])
# ax4.plot(day_4.time,day_4['count'])
f.subplots_adjust(hspace=0)
f.set_size_inches(10,7)


# In[ ]:



