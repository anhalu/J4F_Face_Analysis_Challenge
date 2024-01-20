import pandas as pd
data = pd.read_csv('labels.csv')
data = data.applymap(lambda x: x.lower() if type(x) == str else x)

data.to_csv('labels_lower.csv', index=False)

data = pd.read_csv('labels_for_more_training_data (1).csv')
data = data.applymap(lambda x: x.lower() if type(x) == str else x)

data.to_csv('labels_more_lower.csv', index=False)