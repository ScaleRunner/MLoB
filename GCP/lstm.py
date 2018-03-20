import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed, LSTM
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import sklearn
import argparse

import json
from tensorflow.python.lib.io import file_io

def pandas_to_traintestsplit(dataframe, test_split=.3):
    X = np.asarray(dataframe['comment_vect_numeric'])
    y = np.asarray(dataframe[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

    return X_train, X_test, y_train, y_test


def lstm_model(vocabulary, hidden_size=200):
    model = Sequential()
    # Vocabulary = length total unique dict
    # Embedding layer creates word2vec vector
    model.add(Embedding(input_dim=vocabulary, output_dim=hidden_size))
    model.add(LSTM(hidden_size, dropout=.2))
    model.add((Dense(6, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model

'''
Column-wise AUC loss function
with 'naive' threshold at .5
'''


# TODO: Is this really column wise ??  see:  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation
def score_function(y_true, y_predict, threshold=.5):
    y_predict[y_predict > threshold] = 1
    y_predict[y_predict < threshold] = 0

    score = sklearn.metrics.roc_auc_score(y_true, np.exp(y_predict))

    print("score = ", score)
    return np.mean(score)

def main(train_file='processed_train_1000_data.json'):
    # Load data from JSON file.
    with file_io.FileIO(train_file, 'r') as f:
        json_data = json.load(f)
    data = pd.DataFrame.from_dict(json_data)

    vocab_size = len(set([x for l in data['comment_vect_numeric'].values for x in l]))

    X_train, X_test, y_train, y_test = pandas_to_traintestsplit(data)

    X_train = sequence.pad_sequences(X_train, maxlen=200)
    X_test = sequence.pad_sequences(X_test, maxlen=200)

    network = lstm_model(vocab_size)
    network.fit(X_train, y_train, nb_epoch=1, batch_size=32, verbose=2)
    score, accuracy = network.evaluate(X_test, y_test)

    predications = network.predict(X_test)
    print(predications[0])
    print(predications[20])
    print(predications[40])
    print(predications[22])
    print(predications[25])

    print('Test score:', score)
    print('Test accuracy:', accuracy)

    print("average auc score", score_function(y_test, predications))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop('job_dir')
    
    main(**arguments)
