# 각 file_name에서 dis가 500이상인 애들을 잡고 그 x1 좌표에서 +- 50인 인덱스들 다 날리기
import pandas as pd

df = pd.read_csv('20230518submit.csv')

mask = df['y_dis'] >= 500

file_names = df.loc[mask, 'file_name']
point1_x_values = df.loc[mask, 'point1_x']

for file_name, point1_x in zip(file_names, point1_x_values):
    df = df.drop(df[(df['file_name'] == file_name) & (df['point1_x'] >= point1_x - 75) & (
            df['point1_x'] <= point1_x + 75)].index)

mask = df['ratio'] < 0.7

file_names = df.loc[mask, 'file_name']
point1_x_values = df.loc[mask, 'point1_x']

for file_name, point1_x in zip(file_names, point1_x_values):
    df = df.drop(df[(df['file_name'] == file_name) & (df['point1_x'] >= point1_x - 75) & (
            df['point1_x'] <= point1_x + 75)].index)

df.to_csv('post.csv', index=False)
