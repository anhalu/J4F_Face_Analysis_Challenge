import os
import json
import paddleclas
import glob
import pandas as pd
from tqdm import tqdm

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

image2id = json.load(open('name2id.json'))
id2name = {v: k for k, v in image2id.items()}

df = {
    'file_name': [],
    'image_id': [],
    'bbox': [],
    'race': [],
    'age': [],
    'emotion': [],
    'gender': [],
    'skintone': [],
    'masked': []
}

model = paddleclas.PaddleClas(inference_model_dir='param/resnetv2', use_gpu=True, topk=20)
cnt = 0
a = set()
for file in tqdm(glob.glob('/home/anhalu/anhalu-data/AI_Hackathon/image_new/data256_new/*.jpg')):
    kq = {}
    name_file = os.path.basename(file)
    result = model.predict(input_data=file)
    res = next(result)
    res_label = [idx2label[idx] for idx in res[0]['class_ids']]

    score_label = res[0]['scores']

    for label, score in zip(res_label, score_label):
        for name, list_key in group.items():
            if name not in kq and label in list_key and score > 0.7:
                kq[name] = label

    for k, v in group.items():
        if k not in kq:
            kq[k] = v[0]
            a.add(k)

    df['file_name'].append(name_file)
    df['image_id'].append("")
    df['bbox'].append([])
    df['race'].append(kq['race'])
    df['age'].append(kq['age'])
    df['emotion'].append(kq['emotion'])
    df['gender'].append(kq['gender'])
    df['masked'].append(kq['masked'])
    df['skintone'].append(kq['skintone'])

df = pd.DataFrame(df)
# df = df.sort_values('image_id')

df.to_csv('prediction.csv')
print(a)
