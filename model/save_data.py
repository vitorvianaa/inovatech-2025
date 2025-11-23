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



dfA = pd.read_csv('dataset_hands/dataset_training_A.csv')
dfB = pd.read_csv('dataset_hands/dataset_training_B.csv')
dfC = pd.read_csv('dataset_hands/dataset_training_C.csv')
dfD = pd.read_csv('dataset_hands/dataset_training_D.csv')
dfE = pd.read_csv('dataset_hands/dataset_training_E.csv')
dfF = pd.read_csv('dataset_hands/dataset_training_F.csv')
dfG = pd.read_csv('dataset_hands/dataset_training_G.csv')
dfH = pd.read_csv('dataset_hands/dataset_training_H.csv')
dfI = pd.read_csv('dataset_hands/dataset_training_I.csv')
dfJ = pd.read_csv('dataset_hands/dataset_training_J.csv')
dfK = pd.read_csv('dataset_hands/dataset_training_K.csv')
dfL = pd.read_csv('dataset_hands/dataset_training_L.csv')
dfM = pd.read_csv('dataset_hands/dataset_training_M.csv')
dfN = pd.read_csv('dataset_hands/dataset_training_N.csv')
dfO = pd.read_csv('dataset_hands/dataset_training_O.csv')
dfP = pd.read_csv('dataset_hands/dataset_training_P.csv')
dfQ = pd.read_csv('dataset_hands/dataset_training_Q.csv')
dfR = pd.read_csv('dataset_hands/dataset_training_R.csv')
dfS = pd.read_csv('dataset_hands/dataset_training_S.csv')
dfT = pd.read_csv('dataset_hands/dataset_training_T.csv')
dfU = pd.read_csv('dataset_hands/dataset_training_U.csv')
dfV = pd.read_csv('dataset_hands/dataset_training_V.csv')
dfW = pd.read_csv('dataset_hands/dataset_training_W.csv')
dfX = pd.read_csv('dataset_hands/dataset_training_X.csv')
dfY = pd.read_csv('dataset_hands/dataset_training_Y.csv')
dfZ = pd.read_csv('dataset_hands/dataset_training_Z.csv')

df_combinado = pd.concat([dfA, dfB, dfC, dfD, dfE, dfF, dfG, dfH, dfI, dfJ, dfH, dfI, dfJ, dfK, dfL, dfM, dfN, dfO, dfP, dfQ, dfR, dfS, dfT, dfU, dfV, dfW, dfX, dfY, dfZ], ignore_index=True)

df_combinado.to_csv('dataset_hands/dataset_training.csv', index=False)