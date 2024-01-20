import glob
import os.path
import json
import pandas as pd
from deepface import DeepFace
import cv2
from tqdm import tqdm
folder_image = '/home/anhalu/anhalu-data/AI_Hackathon/private_test/private_test_data/*.jpg'

df = pd.DataFrame(columns=['file_name', 'image_id', 'bbox', 'race', 'age', 'emotion', 'gender', 'skintone', 'masked'])

image2id = json.load(open('private_test/nam2id.json'))
noface_private = []
for i in tqdm(glob.glob(folder_image)):
    img = cv2.imread(i)
    pred = DeepFace.extract_faces(img, detector_backend='retinaface')
    name = os.path.basename(i)
    flag_save = False
    if not os.path.exists(f'/home/anhalu/anhalu-data/AI_Hackathon/private_test/data256/{i}'):
        flag_save = True
    if len(pred) == 0:
        print(pred)
        print("use ssd")
        pred = DeepFace.extract_faces(img, detector_backend='ssd')
        if len(pred) == 0:
            noface_private.append(name)
            continue
    pred = pred[0]
    x, y, w, h = pred['facial_area']['x'], pred['facial_area']['y'], pred['facial_area']['w'], pred['facial_area']['h']
    crop = img[y:y + h, x:x + w]
    if flag_save:
        crop = cv2.resize(crop, (256, 256), cv2.INTER_LINEAR)
        cv2.imwrite(f'/home/anhalu/anhalu-data/AI_Hackathon/private_test/data256/{i}', crop)
    bbox = [x, y, w, h]
    race = ""
    age = ''
    emotion = ''
    gender = ''
    skintone = ''
    masked = ''
    new_row = {'file_name': name, 'bbox': bbox, 'image_id': image2id[name], 'race': race, 'age': age,
               'emotion': emotion, 'gender': gender, 'skintone': skintone, 'masked': masked
               }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df_sorted = df.sort_values(by='image_id')
df_sorted.to_csv('private_test_sample.csv', index=False)

with open('noface_private.txt') as f:
    f.write('\n'.join(noface_private))
