import glob
import pickle
from os.path import join
import datetime

import pandas as pd

from config import cfg

import warnings
warnings.filterwarnings(action='ignore')

with open(join(cfg.RESULT_PATH, '0601result.pkl'), "rb") as f:
    data = pickle.load(f)

results = pd.read_csv(cfg.SAMPLE_SUBMISSION_PATH)
test_img_paths = sorted(glob.glob(cfg.TEST_IMAGE))

for idx in range(len(test_img_paths)):
    prediction = data[idx]
    filename = test_img_paths[idx].split('\\')[-1]
    for i in range(34):
        if prediction[i].shape[0] != 0:
            label = i
            for j in range(prediction[i].shape[0]):
                x1, y1, x2, y2 = prediction[i][j][0], prediction[i][j][1], prediction[i][j][2], prediction[i][j][3]
                score = prediction[i][j][4]
                results = results.append({
                    "file_name": filename,
                    "class_id": label,
                    "confidence": score,
                    "point1_x": x1, "point1_y": y1,
                    "point2_x": x2, "point2_y": y1,
                    "point3_x": x2, "point3_y": y2,
                    "point4_x": x1, "point4_y": y2,
                }, ignore_index=True)

current_time = datetime.datetime.now()
file_name = current_time.strftime("%Y%m%d%H%M")
results.to_csv(join(cfg.SUBMISSION_PATH, f'{file_name}submission.csv'), index=False)

