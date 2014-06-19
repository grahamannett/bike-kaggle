# import main

# data = 
# data =nicedata()

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn import tree, svm


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

def dec_tree_reg(df, test):
    dt = tree.ExtraTreeRegressor()
    #set target, train and test. train and test must have same number of features
    target = df['count']
    train  = df[['time','holiday','season','temp','atemp','windspeed','weather','humidity']]
    test   = test2[['time','holiday','season','temp','atemp','windspeed','weather','humidity']]
    dt.fit(train,target)


    predicted_probs = dt.predict(test)
    predicted_probs = pd.Series(predicted_probs)
    predicted_probs = predicted_probs.map(lambda x: int(x))

    keep = pd.read_csv('data/test.csv')
    keep = keep['datetime']
    #save to file
    submit = pd.concat([keep,predicted_probs],axis=1)
    # print(forest.feature_importances_)
    submit.columns=['datetime','count']
    submit.to_csv('data/submissiondtree.csv',index=False)

dec_tree_reg(data,test2)