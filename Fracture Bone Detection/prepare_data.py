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
        df= pd.DataFrame(columns=['filename','label', 'coords'])
        for file in files:
            text_file = file.replace(".jpg", ".txt").replace("images", "labels")

            with open(text_file, mode="r") as f:
                        lines = f.readlines()
                        for line in lines:
                            values = [float(value) for value in line.split()]
                            label = int(values[0])
                            coords = values[1:]
                            row_data = [file]
                            row_data.append(label)
                            row_data.append(coords)
                            df.loc[len(df.index)] = row_data

        df.to_csv(file_dataname, sep=';', encoding='utf-8', index=False)

    df_labels = df[['filename','label']].copy()
    df_boxes = df[['filename','coords']].copy()

    df_labels['cat_label'] = '-'
    fracture_names= ['elbow positive', 'fingers positive', 
        'forearm fracture', 'humerus fracture', 
        'humerus', 'shoulder fracture', 'wrist positive']
    for i in range(len(fracture_names)):
        df_labels.loc[df_labels['label']==i,'cat_label'] = fracture_names[i]
    df_labels.drop('label', axis=1, inplace=True)
    df_labels = pd.get_dummies(df_labels, columns=['cat_label'], dtype='int')
    
    df_cols = df_labels.columns
    df_cols = [x.replace('cat_label_', '') for x in df_cols]
    df_labels.columns = df_cols

    cols = ['filename']
    for frac_name in fracture_names:
         cols.append(frac_name)
         if not(frac_name in df_cols):
              df_labels[frac_name]=0
    df_labels = df_labels[cols]              

    df_labels.reset_index(drop=True, inplace=True)
    df_boxes.reset_index(drop=True, inplace=True)
    return df_labels, df_boxes