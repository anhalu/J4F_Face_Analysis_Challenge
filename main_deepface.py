import time
import os
import glob
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align
from tqdm import tqdm
import pandas as pd
import json

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                   allowed_modules=['detection', 'landmark_3d_68'])
# app.prepare(ctx_id=0, det_size=(320, 320))
app.prepare(ctx_id=0, det_thresh=0.1)

noface = []

df = pd.DataFrame(columns=['file_name', 'image_id', 'bbox', 'race', 'age', 'emotion', 'gender', 'skintone', 'masked'])

image2id = json.load(open('private_test/nam2id.json'))

for file in tqdm(glob.glob('/home/anhalu/anhalu-data/AI_Hackathon/private_test/private_test_data/*.jpg')):
    name = os.path.basename(file)
    # if os.path.exists(f"/home/anhalu/anhalu-data/AI_Hackathon/private_test/data256/{name}"):
    #     continue
    # img = ins_get_image(file)
    # cv2.imwrite("face_test.jpg", img)
    img = cv2.imread(file)
    start_time = time.time()
    faces = app.get(img)

    bbox = None
    if len(faces) == 0:
        print("NO FACE in ", name)
        crop = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
        cv2.imwrite(f"/home/anhalu/anhalu-data/AI_Hackathon/private_test/data256/{name}", crop)
        noface.append(name)
        bbox = [0, 0, img.shape[1], img.shape[0]]
    conf = 0
    for i, face in enumerate(faces):
        pitch, yaw, roll = face['pose']
        x1, y1, x2, y2 = list(face['bbox'])
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        # print(x1, y1, x2, y2)

        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=256)
        # if yaw < -54 or yaw > 54 or pitch < -45 or pitch > 30:
        #     print("Invalid ", name)
        # cv2.imwrite(f"/home/anhalu/anhalu-data/AI_Hackathon/image_old/data256/{name}", aimg)

        if conf < face['det_score']:
            conf = face['det_score']
            # cv2.imwrite(f"/home/anhalu/anhalu-data/AI_Hackathon/private_test/data256/{name}", aimg)
            bbox = [x1, y1, x2 - x1, y2 - y2]  # x, y, w , h

    # bbox = [x, y, w, h]
    race = 'mongoloid'
    age = '20-30s'
    emotion = 'happiness'
    gender = 'female'
    skintone = 'light'
    masked = 'unmasked'
    new_row = {'file_name': name, 'bbox': bbox, 'image_id': image2id[name], 'race': race, 'age': age,
               'emotion': emotion, 'gender': gender, 'skintone': skintone, 'masked': masked
               }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df_sorted = df.sort_values(by='image_id')
df_sorted.to_csv('private_test_sample.csv', index=False)

# print(time.time() - start_time)

with open('noface.txt', 'w') as f:
    f.write('\n'.join(noface))

    # 1193565.jpg 1281272.jpg
