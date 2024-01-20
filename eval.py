import pandas as pd

labels_df = pd.read_csv('/home/anhalu/anhalu-data/AI_Hackathon/image_new/new.csv')
predictions_df = pd.read_csv('prediction.csv')

merged_df = pd.merge(labels_df, predictions_df, on='file_name')
merged_df['age_x'] = merged_df['age_x'].str.lower()
merged_df['race_x'] = merged_df['race_x'].str.lower()
merged_df['emotion_x'] = merged_df['emotion_x'].str.lower()
merged_df['gender_x'] = merged_df['gender_x'].str.lower()
merged_df['skintone_x'] = merged_df['skintone_x'].str.lower()
merged_df['masked_x'] = merged_df['masked_x'].str.lower()

merged_df.to_csv("merged_df.csv", index=False)
l = set()
colums = ['race', 'age', 'emotion', 'gender', 'skintone', 'masked']
for name in colums:
    correct_predictions = merged_df[f'{name}_x'] == merged_df[f'{name}_y']
    for idx, i in enumerate(correct_predictions):
        if not i:
            l.add(idx)

    print(merged_df[correct_predictions == False][f'{name}_x'].value_counts())
    # print(merged_df[correct_predictions == False][f'{name}_y'].value_counts())
    accuracy = correct_predictions.sum() / len(correct_predictions)
    print(f'Accuracy {name}: {accuracy * 100:.2f}%')

l = sorted(list(l))
miss = labels_df.iloc[l]
miss.to_csv("miss.csv", index=False)