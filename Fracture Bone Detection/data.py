import glob
import os
import pandas as pd

def prepare_data(root, mode):
    file_dataname = f'./data/{mode}/data_{mode}.csv'
    if os.path.exists(file_dataname):
          df = pd.read_csv(file_dataname, delimiter=';')
          print(f'data_{mode}.csv' +' already exists')
    else:
        files = sorted(glob.glob(os.path.join(root, mode) + '/images/*.jpg'))
        df = pd.DataFrame(columns=['label', 'a1', 'a2', 'a3',
                                'a4', 'a5', 'a6', 'a7', 'a8', 'filename'])
        for file in files:
            text_file = file.replace(".jpg", ".txt").replace("images", "labels")

            with open(text_file, mode="r") as f:
                        lines = f.readlines()
                        for line in lines:
                            values = [float(value) for value in line.split()]
                            #label = int(values[0])
                            #coords = values[1:]
                            values[0] = int(values[0])
                            values = values[:9]
                            row_data = values
                            row_data.append(file)
                            df.loc[len(df.index)] = row_data

        df.to_csv(file_dataname, sep=';', encoding='utf-8', index=False)
    return df

root = './data'
mode = 'train'

df = prepare_data(root, mode)
print(df)