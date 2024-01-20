import os.path

import pandas as pd


dic = {
    'kid': 0,
    'baby': 1,
    'teenager': 2,
    '20-30s': 3,
    '40-50s': 4,
    'senior': 5,
    'caucasian': 6,
    'mongoloid': 7,
    'negroid': 8,
    'unmasked': 9,
    'masked': 10,
    'mid-light': 11,
    'light': 12,
    'mid-dark': 13,
    'dark': 14,
    'neutral': 15,
    'happiness': 16,
    'anger': 17,
    'surprise': 18,
    'fear': 19,
    'sadness': 20,
    'disgust': 21,
    'male': 22,
    'female': 23
}

idx2label = {v: k for k, v in dic.items()}

miss = []

data = pd.read_csv('image_new/new.csv')
data = data.applymap(lambda x: x.lower() if type(x) == str else x)
all = []
for i in data.itertuples():
    if not os.path.exists(f'image_new/data256_new/{i.file_name}'):
        print("?")
        miss.append(f'image_new/face_new/{i.file_name}')
    label = ['0'] * 24
    label[dic[i.race]] = '1'
    label[dic[i.age]] = '1'
    label[dic[i.emotion]] = '1'
    label[dic[i.gender]] = '1'
    label[dic[i.skintone]] = '1'
    label[dic[i.masked]] = '1'
    label = ','.join(label)
    all.append(f'{i.file_name}\t{label}')


data = pd.read_csv('image_old/labels.csv')
data = data.applymap(lambda x: x.lower() if type(x) == str else x)
val = []
cnt = 0
for i in data.itertuples():
    if not os.path.exists(f'image_old/data256_old/{i.file_name}'):
        print("??")
        miss.append(f'image_old/face_old/{i.file_name}')

    label = ['0'] * 24
    label[dic[i.race]] = '1'
    label[dic[i.age]] = '1'
    label[dic[i.emotion]] = '1'
    label[dic[i.gender]] = '1'
    label[dic[i.skintone]] = '1'
    label[dic[i.masked]] = '1'
    label = ','.join(label)
    if cnt < 1500:
        val.append(f'{i.file_name}\t{label}')
    else:
        all.append(f'{i.file_name}\t{label}')
    cnt += 1


with open('labels.txt', 'w') as f:
    f.write('\n'.join(all))

with open('val.txt', 'w') as f:
    f.write('\n'.join(val))

print(len(miss))

with open('miss.txt', 'w') as f:
    f.write('\n'.join(miss))