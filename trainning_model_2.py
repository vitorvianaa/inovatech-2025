from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pandas as pd

# load dataset
df = pd.read_csv("dataset_training.csv")
X = df.drop('label', axis=1).values
y = df['label'].values

# normilized data
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# slipt data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# create model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# traning
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# validate
loss, acc = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {acc:.2f}")

model.save('first_model.h5')
