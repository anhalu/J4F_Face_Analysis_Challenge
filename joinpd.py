import pandas as pd

df_an = pd.read_csv('/home/anhalu/anhalu-data/AI_Hackathon/answer.csv')
df_agegener = pd.read_csv('/home/anhalu/anhalu-data/AI_Hackathon/genderage.csv')

df = pd.merge(df_an, df_agegener, left_on='file_name', right_on='name', how='left')
df['age'] = df['age_y']
idx = df['age'].isnull()
df.loc[idx, 'age'] = df.loc[idx, 'age_x']

df['gender'] = df['gender_y']
idx = df['gender'].isnull()
df.loc[idx, 'gender'] = df.loc[idx, 'gender_x']

df_con = df[['file_name', 'bbox', 'image_id', 'race', 'age', 'emotion', 'gender', 'skintone', 'masked']]

df_con.to_csv('save_v2.csv', index=False)
