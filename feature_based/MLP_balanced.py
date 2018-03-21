import os
import nltk
from keras.models import Sequential, optimizers
from keras.layers import Dense, Dropout, BatchNormalization
import pandas as pd
from feature_based.preprocess.features import extract_features, extract_labels, split_data, normalize_features
import numpy as np
import sklearn
from tqdm import tqdm

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
    np.savetxt("../features_train.csv", features, delimiter=',') if train else np.savetxt("../features_test.csv",
                                                                                          features, delimiter=',')

    labels = extract_labels(df) if train else None

    return features, np.array(labels), ids


def create_model(n_features, output_size):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_features,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_size, activation='softmax')
    ])

    # Multiclass uses categorical cross-entropy
    optimizer = optimizers.Adam(lr=1e-5)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(features, labels, model, n_epochs=100, batch_size=256, verbose=2, percentage_of_class=0.75):
    positive_samples = features[labels == 1, :]
    positive_labels = labels[labels == 1]
    negative_samples = features[labels == 0, :]
    negative_labels = labels[labels == 0]

    samples_per_epoch = len(positive_samples) // 2

    for _ in range(n_epochs):
        pos_idxs = np.random.choice(np.arange(positive_samples.shape[0]),
                                    size=int(samples_per_epoch * percentage_of_class), replace=False)
        neg_idxs = np.random.choice(np.arange(negative_samples.shape[0]),
                                    size=int(samples_per_epoch * (1 - percentage_of_class)), replace=False)

        features_batch = np.concatenate((positive_samples[pos_idxs], negative_samples[neg_idxs]))
        labels_batch = np.concatenate((positive_labels[pos_idxs], negative_labels[neg_idxs]))
        # This is not sanitary and should be changed!
        features_train, features_validation, labels_train, labels_validation = split_data(features_batch, labels_batch)

        # for _ in range(n_epochs):
        #     data_train = []
        model.fit(features_train, labels_train,
                  epochs=1,
                  batch_size=batch_size,
                  verbose=verbose,
                  validation_data=(features_validation, labels_validation),
                  shuffle=True)
    print(model.evaluate(features_batch, labels_batch))


def score_function(y_true, y_predict):
    """
    :param y_true:
    :param y_predict:
    :return: Mean averaged column wise AUC score
    """
    scores = []
    for i in range(4):
        scores.append(sklearn.metrics.roc_auc_score(y_true[:, i], y_predict[:, i]))
    print("score = ", np.mean(scores))
    return np.mean(scores)


def main():
    print("Loading Training Data")
    # features_train, labels_train = load_data('../processed_train_data.csv')
    # features_test, labels_test = load_data('../processed_test_data.csv')
    features_train, labels_train, _ = load_data('../data/train.csv')

    n_features = features_train.shape[1]
    output_size = labels_train.shape[1]
    print("Training has nan:", np.isnan(features_train).any())

    models = []
    for i in range(output_size):
        cur_model = create_model(n_features, 1)
        train_model(features_train, labels_train[:, i], cur_model, n_epochs=100)
        models.append(cur_model)

    print("Loading Test Data")
    features_test, _, ids = load_data('../data/test.csv', train=False)
    print("Testing has nan:", np.isnan(features_test).any())

    # Testing
    predictions = np.array([], dtype=np.int8).reshape(features_train.shape[0], 0)
    # predictions = model.predict(features_test, verbose=2)

    np.savetxt("../predictions.csv", predictions, delimiter=',')

    with open("../predictions_submission.csv", "w") as f:
        f.write("id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n")
        for i, preds in enumerate(predictions):
            f.write("{},{},{},{},{},{},{}\n".format(ids[i], *preds))


if __name__ == "__main__":
    main()
