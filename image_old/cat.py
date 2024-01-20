from tqdm import tqdm

import cv2
import pandas as pd

label = pd.read_csv('new.csv')

for i, row in tqdm(label.iterrows()):

    img = cv2.imread('data/' + row['file_name'])
    x, y, w, h = row['bbox'][1:-1].split(',')
    x = int(float(x))
    y = int(float(y))
    w = int(float(w))
    h = int(float(h))
    crop = img[y:y+h, x:x+w]
    cv2.imwrite('face_new/' + row['file_name'], crop)


