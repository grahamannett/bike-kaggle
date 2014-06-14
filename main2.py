import pandas as pd

data = pd.read_csv('sampleSubmission.csv')
x = 's'
data['datetime'] = data['datetime'].str.replace(':00', ':00:00')
data['datetime'] = data['datetime'].replace('011/', '011-0')
i = 0


data.datetime.replace('20', 'xxx')

print(data)


data.to_csv('sampleSubmission2.csv', index=False)
d = '5'
