import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
type_data = os.getenv('TYPE_DATA')
def save_data(dataset: list, label: str):
    rows = []
    for data in dataset:
        rows.append(data['landmarks'] + [data['label']])

    columns = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(f'dataset_{type_data}_{label}.csv', index=False)
    print('Dataset salvo...')



df1 = pd.read_csv('dataset_training_A.csv')
df2 = pd.read_csv('dataset_training_B.csv')
df3 = pd.read_csv('dataset_training_C.csv')
df4 = pd.read_csv('dataset_training_D.csv')
df5 = pd.read_csv('dataset_training_E.csv')

df_combinado = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

df_combinado.to_csv('dataset_training.csv', index=False)