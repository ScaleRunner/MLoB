import os
import nltk
import pandas as pd
from feature_based.preprocess.features import extract_features, extract_labels, split_data, normalize_features
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import sklearn


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Remove Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def load_data(path, train=True, size=None):
    df = pd.read_csv(path, nrows=size, encoding='utf-8')
    df.dropna()
    ids = df['id']
    features = extract_features(df)
    features = np.nan_to_num(features)
    features = normalize_features(features)
    np.savetxt("../features_train.csv", features, delimiter=',') if train else np.savetxt("../features_test.csv", features, delimiter=',')

    labels = extract_labels(df) if train else None

    return features, np.array(labels), ids


def score_function(y_true, y_predict):
    """
    :param y_true:
    :param y_predict:
    :return: Mean averaged column wise AUC score
    """
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(sklearn.metrics.roc_auc_score(y_true[:, i], y_predict[:, i]))
    print("score = ", np.mean(scores))
    return np.mean(scores)


def main():
    print("Loading Training Data")
    features_train, labels_train, _ = load_data('../data/train.csv')
    print("Training has nan:", np.isnan(features_train).any())

    # Training
    clf = OneVsRestClassifier(SVC(kernel='linear'))
    clf.fit(features_train, labels_train)

    # Check AUC score for training
    predictions_train = clf.predict_proba(features_train)
    print("AUC:", score_function(labels_train, predictions_train))

    # Make Predictions
    print("Loading Test Data")
    features_test, _, ids = load_data('../data/test.csv', train=False)
    print("Testing has nan:", np.isnan(features_test).any())

    predictions_test = clf.predict_proba(features_test)
    np.savetxt("../predictions.csv", predictions_test, delimiter=',')

    with open("../predictions_submission.csv", "w") as f:
        f.write("id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n")
        for i, preds in enumerate(predictions_test):
            f.write("{},{},{},{},{},{},{}\n".format(ids[i], *preds))


if __name__ == "__main__":
    main()
