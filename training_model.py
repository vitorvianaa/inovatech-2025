from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

type_model = os.getenv('TYPE_DATA')
current_label = os.getenv('CURRENT_LABEL')

df = pd.read_csv(f'dataset_{type_model}.csv')
# colunad do landmark
X = df.drop('label', axis=1).values

# labels
y = df['label'].values

# letra -> numero
encoder = LabelEncoder()
y_encode = encoder.fit_transform(y)

# dado pra rede neural
y_categorical = to_categorical(y_encode)


model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_categorical, epochs=100, validation_split=0.2)

model.save('first_model.h5')