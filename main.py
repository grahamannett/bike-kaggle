# guide
# github.com/chrisclark/PythonForDataScience/blob/master/makeSubmission.py
# from tutorial:
# blog.kaggle.com/2012/07/02/up-and-running-with-python-my-first-kaggle-entry

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.linear_model import BayesianRidge, LinearRegression
import statsmodels.formula.api as smf

#remove warning
pd.options.mode.chained_assignment = None


def nicedata(df):

    #split datetime into 2 columns
    time = [item.split(' ')[1] for item in df['datetime']]
    date = pd.Series([item.split(' ')[0] for item in df['datetime']])
    #rename columns
    df['datetime'].update(date)
    df['time'] = time
    df.rename(columns={'datetime': 'date'}, inplace=True)
    #reorganize columns
    col = df.columns.tolist()
    col = col[-1:]+col[:-1]
    col[0], col[1] = col[1], col[0]
    df = df[col]
    #remove :00:00 on every time
    df['time'] = df['time'].str.replace(':00:00', '')
    return df

def submitdata(predictions,test2):

    keep = pd.read_csv('data/test.csv')
    keep = keep['datetime']
    #save to file
    predicted_probs = 'z'
    submit = pd.concat([keep, predicted_probs], axis=1)
    submit.columns = ['datetime', 'count']
    submit.to_csv('data/submission.csv', index=False)

data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

data = nicedata(data)
test2 = nicedata(test)
#print(test.head(3))

def rforest(df, test, est=5):
    forest = RandomForestRegressor(n_estimators=est)
    #set target, train and test. train and test must have same number of features
    target = df['count'].values
    train  = df[['time','holiday','season','temp','atemp','windspeed','weather','humidity']].values
    test   = test2[['time','holiday','season','temp','atemp','windspeed','weather','humidity']].values
    forest.fit(train,target)


    predicted_probs = forest.predict(test)
    predicted_probs = pd.Series(predicted_probs)
    predicted_probs = predicted_probs.map(lambda x: int(x))

    keep = pd.read_csv('data/test.csv')
    keep = keep['datetime']
    #save to file
    submit = pd.concat([keep,predicted_probs],axis=1)
    print(forest.feature_importances_)
    submit.columns=['datetime','count']
    submit.to_csv('data/submission.csv',index=False)

# rforest(data, test2 ,1000)

def regfit(df,test):
    y               = df['count']
    X               = df[['time','season','temp']]
    results         = smf.ols('count ~ season +holiday + time+temp +atemp + windspeed', data=df).fit()
    print results.summary()
    test            = test2[['time','season','temp','windspeed','atemp','holiday']]
    
    
    #predict
    predicted_probs = results.predict(test)
    #save to file
    keep            = pd.read_csv('data/test.csv')
    keep            = keep['datetime']
    predicted_probs = pd.Series(predicted_probs)
    submit          = pd.concat([keep,predicted_probs],axis=1)
    submit.columns  =['datetime','count']
    predicted_probs.to_csv('data/submissionrlm.csv',index=False)

# regfit(data,test2)

#ridge regression

def ridreg(df,test):
    clf = BayesianRidge()
    
    target = df['count']
    train  = df[['time','temp']]
    test   = test2[['time','temp']]

    clf.fit(train,target)
    final = []
    print(test.head(3))
    for i, row in enumerate(test.values):
        y=[]
        for x in row:
            x= float(x)
            y.append(x)
            # print(x)
            final.append(y)
    predicted_probs= clf.predict(final)
    # print(predicted_probs.shape)
    # predicted_probs = pd.Series(predicted_probs)
    # predicted_probs = predicted_probs.map(lambda x: int(x))

    keep = pd.read_csv('data/test.csv')
    keep = keep['datetime']
    # #save to file
    predicted_probs= pd.DataFrame(predicted_probs)
    print(predicted_probs.head(3))
    predicted_probs.to_csv('data/submission3.csv',index=False)


# ridreg(data,test2)

def suppvecreg(df,test2):
    target = df['count']
    train  = df[['time','temp','humidity']]
    test   = test2[['time','temp','humidity']]

    clf2 = svm.SVR(degree=5,probability=True)
    clf2.fit(train,target)
    # 
    p123= clf2.predict(test)
    p123.to_csv('data/submission4.csv',index=False)


# suppvecreg(data,test2)

def gradb(df, test):
    return 5

#plottign below
toplot = data[['time','count']]
timeplot = pd.Series(list(toplot['time']))
countplot= pd.Series(list(toplot['count']))

print(toplot.shape)
toplot= (toplot.groupby('time',sort=True).sum())/270
# toplot = toplot.grouby(toplot.pd.tseries.index.hour)
# print(toplot.head(30))
# plt.figure()
# plt.plot(toplot)
# pd.tools.plotting.scatter_matrix(data)
# plt.show()


