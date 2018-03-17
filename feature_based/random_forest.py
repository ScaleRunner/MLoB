import os

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

from preprocess import load_and_process_data, load_data, split_data


def create_random_forest(max_depth=5, min_samples_leaf=20, verbose=False):
    r_forest = OneVsRestClassifier(RandomForestClassifier(n_estimators=40, max_depth=max_depth, min_samples_leaf=min_samples_leaf, verbose=verbose), n_jobs=5)
    
    return r_forest


def train_model(features, labels, model, n_epochs=50, batch_size=32, verbose=2):
    features_train, features_validation, labels_train, labels_validation = split_data(features, labels)

    # The labels should be ints, they are floats?
    labels_train = ((labels_train.astype(np.int)))
    labels_validation = labels_validation.astype(np.int)

    model = model.fit(features_train, labels_train)
    score = model.score(features_validation, labels_validation)
    print(score)
    return model


def main():

    features_train, labels_train = load_data(size=50000)
    
    n_features = features_train.shape[1]
    output_size = labels_train.shape[1]

    model = create_random_forest(max_depth=1, min_samples_leaf=1, verbose=True)

    # Training
    train_model(features_train, labels_train, model)

    # Testing
    print("Loading Test Data")
    features_test, _ = load_and_process_data('../data/test.csv', train=False)
    predictions = model.predict(features_test)

    np.savetxt("../predictions.csv", predictions, delimiter=',')


if __name__ == "__main__":
    main()
