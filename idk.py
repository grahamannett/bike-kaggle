def rforest(df, test, est=5):
    forest = RandomForestRegressor(n_estimators=est)
    #set target, train and test. train and test must have same number of features
    target = df['count'].values
    train  = df[['holiday','season','temp','atemp','windspeed','weather','humidity']].values
    test   = test2[['holiday','season','temp','atemp','windspeed','weather','humidity']].values
    forest.fit(train,target)


    predicted_probs = forest.predict(test)
    predicted_probs = pd.Series(predicted_probs)
    predicted_probs = predicted_probs.map(lambda x: int(x))

    keep = pd.read_csv('data/test.csv')
    t = keep[keep['time']=='01']
    t = t['datetime']
    
    #save to file
    submit = pd.concat([t,predicted_probs],axis=1)
    print(forest.feature_importances_)
    submit.columns=['datetime','count']
    submit.to_csv('data/submission.csv',index=False)

# rforest(data, test2 ,1000)

def regfit(df,test):
    y = df['count']
    X  = df[['time','season','temp']]
    results = smf.ols('count ~ season +holiday + time+temp +atemp + windspeed', data=df).fit()
    print results.summary()
    test   = test2[['time','season','temp','windspeed','atemp','holiday']]


    #predict
    predicted_probs = results.predict(test)
    #save to file
    keep = pd.read_csv('data/test.csv')
    keep = keep['datetime']
    predicted_probs = pd.Series(predicted_probs)
    submit = pd.concat([keep,predicted_probs],axis=1)
    submit.columns=['datetime','count']
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

