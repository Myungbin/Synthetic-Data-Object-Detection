import os

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\MB_Project\project\Competition\VISOL\data\submission\202306011519submission.csv')


def visual_box(n):
    points = df.loc[:, 'point1_x':'point4_y'].values[n].reshape(-1, 2)
    image_path = df['file_name'].values[n]
    image = plt.imread(os.path.join('data', 'test', image_path))
    fig, ax = plt.subplots()
    ax.imshow(image)
    bbox = plt.Polygon(points, fill=None, edgecolor='red')
    ax.add_patch(bbox)
    plt.show()


visual_box(2)