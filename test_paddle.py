import os

import paddleclas
import glob
import pandas as pd
from tqdm import tqdm
import cv2

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
group = {
    'age': ['20-30s', 'kid', 'baby', 'teenager', '40-50s', 'senior'],
    'gender': ['female', 'male'],
    'masked': ['unmasked', 'masked'],
    'race': ['mongoloid', 'caucasian', 'negroid'],
    'skintone': ['light', 'mid-light', 'mid-dark',  'dark'],
    'emotion': ['happiness', 'neutral', 'anger', 'surprise', 'fear', 'sadness', 'disgust']
}
# list_res = []

df = pd.read_csv('private_test_sample.csv')
model = paddleclas.PaddleClas(inference_model_dir='param/final', use_gpu=True, topk=20)
for file in tqdm(glob.glob('/home/anhalu/anhalu-data/AI_Hackathon/private_test/data256/*.jpg')):
    kq = {}
    name_file = os.path.basename(file)
    img = cv2.imread(file)
    result = model.predict(img)
    res = next(result)

    # print(res[0]['class_ids'])
    res_label = [idx2label[idx] for idx in res[0]['class_ids']]
    score_label = res[0]['scores']

    for label, score in zip(res_label, score_label):
        for name, list_key in group.items():
            if name not in kq and label in list_key and score > 0.5:
                kq[name] = label

    for k, v in group.items():
        if k not in kq:
            kq[k] = v[0]

    for k, v in kq.items():
        if not os.path.isdir(f'save_res/{k}/{v}'):
            os.mkdir(f'save_res/{k}/{v}')

        cv2.imwrite(f'save_res/{k}/{v}/{name_file}', img)

    df.loc[df['file_name'] == name_file, 'age'] = kq['age']
    df.loc[df['file_name'] == name_file, 'gender'] = kq['gender']
    df.loc[df['file_name'] == name_file, 'masked'] = kq['masked']
    df.loc[df['file_name'] == name_file, 'race'] = kq['race']
    df.loc[df['file_name'] == name_file, 'skintone'] = kq['skintone']
    df.loc[df['file_name'] == name_file, 'emotion'] = kq['emotion']

df = df.drop_duplicates(subset=['file_name', 'bbox'])
df.to_csv('private_test_answer.csv', index=False)

    # list_res.append(kq)
#     print(kq)
#     break
#
# df = pd.DataFrame.from_records(list_res)
# print(df.head())
