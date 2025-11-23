from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import pickle 


df = pd.read_csv(f'dataset_hands/dataset_training.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

encoder = LabelEncoder()
y_encode = encoder.fit_transform(y)
y_categorical = to_categorical(y_encode)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, shuffle=True
)

# model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.2), 
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# save the best and stop if dont better
checkpoint = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

print("melhor acurácia de validação:", max(history.history['val_accuracy']))