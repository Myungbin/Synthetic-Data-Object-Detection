import pandas as pd

df = pd.read_csv('../../final.csv')


def rename_file(filename):
    new_filename = filename.replace(".json", ".png")
    return new_filename


df['file_name'] = df['file_name'].apply(rename_file)


def remove_prefix(filename):
    prefix = "tmp_t\\ensemble\\"
    new_filename = filename[len(prefix):]
    return new_filename


df['file_name'] = df['file_name'].apply(remove_prefix)
df.to_csv('final1.csv', index=False)