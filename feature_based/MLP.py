import os
import nltk
from keras.models import Sequential, optimizers
from keras.layers import Dense
import pandas as pd
from preprocess import load_data, load_and_process_data, split_data

import numpy as np

# Remove Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}



def create_model(n_features, output_size):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_features, )),
        Dense(32, activation='relu'),
        Dense(output_size, activation='softmax')
    ])

    # Multiclass uses categorical cross-entropy
    optimizer = optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(features, labels, model, n_epochs=50, batch_size=32, verbose=2):
    features_train, features_validation, labels_train, labels_validation = split_data(features, labels)

    history = model.fit(features_train, labels_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        validation_data=(features_validation, labels_validation),
                        shuffle=True)

    score = model.evaluate(features_validation, labels_validation, batch_size=batch_size)
    print(score)

    return history


def main():
    print("Loading Training Data")
    # features_train, labels_train = load_data('../processed_train_data.csv')
    # features_test, labels_test = load_data('../processed_test_data.csv')
    features_train, labels_train = load_data('../data/train.csv', size=5000)

    n_features = features_train.shape[1]
    output_size = labels_train.shape[1]

    model = create_model(n_features, output_size)

    # Training
    train_model(features_train, labels_train, model)

    # Testing
    print("Loading Test Data")
    features_test, _ = load_and_process_data('../data/test.csv', train=False)
    predictions = model.predict(features_test, verbose=2)

    np.savetxt("../predictions.csv", predictions, delimiter=',')


if __name__ == "__main__":
    main()
