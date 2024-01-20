import glob
import os

import cv2

from paddleclas import PaddleClas

model = PaddleClas(inference_model_dir='param', topk=20)


for file in glob.glob('image/*.jpg'):
    img = cv2.imread(file)
    name = os.path.basename(file)
    pred = model.predict(img)
    pred = next(pred)
    print(pred)