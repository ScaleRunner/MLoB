from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
from feature_extraction.features import extract_features, train_test_split

# TODO: load data
df = pd.DataFrame()
features = extract_features(df)
labels = df['labels']

features_train, labels_train, features_test, labels_test = train_test_split(features, labels)

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

# Multiclass uses categorical cross-entropy
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(features_train, labels_train,
          epochs=100,
          batch_size=128)
score = model.evaluate(features_test, labels_test, batch_size=128)
