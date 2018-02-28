from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd

## TODO: load data
df = pd.DataFrame()
n_samples = len(df.columns)
n_features = len(df["features"]) # prob not
n_outputs = len(df['labels'])

model = Sequential([
    Dense(64, input_shape=(n_samples, n_features)),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(n_outputs),
    Activation('softmax')  # Maybe, more research needed
])

# Multiclass uses categorical crossentropy
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

## TODO: Training Loop