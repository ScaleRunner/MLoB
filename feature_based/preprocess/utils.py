import numpy as np
import pandas as pd

from .feature_extractor import extract_features, normalize_features, extract_labels
from sklearn.model_selection import train_test_split


def split_data(features, labels, test_percentage=0.1):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, shuffle=True, test_size=test_percentage)

    return np.array(features_train), np.array(features_test), np.array(labels_train, dtype=np.int), np.array(labels_test)


def load_and_process_data(path, train=True, size=None):
    df = pd.read_csv(path, index_col=0, nrows=size, encoding='utf-8')
    df.dropna()
    features = extract_features(df)
    features = normalize_features(features)

    labels = extract_labels(df) if train else None

    return np.array(features), np.array(labels)



def save_data(features, labels, path_name='./'):
    """
        Save the processed features to a csv,
        the feature extraction process takes about 10 minutes.

        Loading a csv about 5s.
    """
    pd.DataFrame(features).to_csv(path_name + 'features.csv', header=None, index=False)
    pd.DataFrame(labels).to_csv(path_name + 'labels.csv', header=None, index=False)

def load_data(path_name='./', size=None):

    features = pd.read_csv('./features.csv', delimiter=',', header=None, nrows=size).values
    labels = pd.read_csv('./labels.csv', delimiter=',', header=None, nrows=size).values

    return features, labels

