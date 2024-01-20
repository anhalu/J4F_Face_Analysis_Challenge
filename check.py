import os

import cv2
with open('miss.txt', 'r') as f:
    data = f.readlines()

for line in data:
    if line.endswith('\n'):
        line = line[:-1]
    name = os.path.basename(line)
    if os.path.exists(f'miss/{name}'):
        continue
    img = cv2.imread(line)
    img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
    cv2.imwrite(f'miss/{name}', img)

