import os
import nltk
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from features import extract_features, extract_labels, split_data
import numpy as np
import csv
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def load_data(path, train=True, size=None):
    df = pd.read_csv(path, index_col=0, nrows=size, encoding='utf-8')
    features = extract_features(df)

    labels = extract_labels(df) if train else None

    return np.array(features), np.array(labels)


def create_model(n_features, output_size):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_features, )),
        Dense(32, activation='relu'),
        Dense(output_size, activation='softmax')
    ])

    # Multiclass uses categorical cross-entropy
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(features, labels, model, verbose=1):
    features_train, features_validation, labels_train, labels_validation = split_data(features, labels)

    history = model.fit(features_train, labels_train,
                        epochs=50,
                        batch_size=512,
                        verbose=verbose,
                        validation_data=(features_validation, labels_validation),
                        shuffle=True)

    score = model.evaluate(features_validation, labels_validation, batch_size=128)
    print(score)

    return history


def main():
    print("Loading Training Data")
    # features_train, labels_train = load_data('../processed_train_data.csv')
    # features_test, labels_test = load_data('../processed_test_data.csv')
    features_train, labels_train = load_data('../data/train.csv', size=100)

    n_features = features_train.shape[1]
    output_size = labels_train.shape[1]

    model = create_model(n_features, output_size)

    # Training
    train_model(features_train, labels_train, model)

    # Testing
    print("Loading Test Data")
    features_test, _ = load_data('../data/test.csv', train=False, size=100)
    predictions = model.predict(features_test, verbose=1)

    np.savetxt("../predictions.csv", predictions, delimiter=',')


if __name__ == "__main__":
    main()
