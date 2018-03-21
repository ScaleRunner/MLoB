from collections import Counter

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, MaxPooling1D, Embedding, Conv1D, LSTM
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import sklearn
from sklearn.utils import class_weight
import h5py
import keras
import collections


def pandas_to_traintestsplit_balanced(dataframe, test_split=.3):
    X = []
    y = []

    for kind in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        data = dataframe[dataframe[kind] == 1]
        x = np.asarray(data['comment_vect_numeric'])
        y_temp = np.asarray(data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])
        X.extend(x)
        y.extend(y_temp)
        print("kind length ", kind, " ", y_temp.shape)
    data = dataframe[(dataframe['toxic'] == 0) & (dataframe['severe_toxic'] == 0) & (dataframe['obscene'] == 0)
                & (dataframe['threat'] == 0) & (dataframe['insult'] == 0) & (dataframe['identity_hate'] == 0)]
    x2 = np.asarray(data['comment_vect_numeric'])
    y_temp2 = np.asarray(data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

    print("lengte 6 nullen: 8000")

    X.extend(x2[:6000])
    y.extend(y_temp2[:6000])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)


    return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

def pandas_to_traintestsplit_unbalanced(dataframe, test_split=.3):
    X = np.asarray(dataframe['comment_vect_numeric'])
    y = np.asarray(dataframe[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    return X_train, X_test, y_train, y_test



def lstm_model(vocabulary, hidden_size=200):
    model = Sequential()
    # Vocabulary = length total unique dict
    # Embedding layer creates word2vec vector
    model.add(Embedding(input_dim=vocabulary, output_dim=hidden_size))
    # model.add(Dropout(0.2))
    # model.add(LSTM(hidden_size)
    # model.add(Dropout(0.2))
    # model.add((Dense(6, activation='sigmoid')))

    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(125))
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model


def score_function(y_true, y_predict):
    '''
    :param y_true:
    :param y_predict:
    :return: Mean averaged column wise AUC score
    '''
    scores = []
    for i in range(4):
        scores.append(sklearn.metrics.roc_auc_score(y_true[:, i], y_predict[:, i]))
    print("score = ", np.mean(scores))
    return np.mean(scores)


def train_lstm(data, padding_length=200,epochs=5, batch_size=64, name='my_model'):
    '''
    :param data: Full pandas dataframe
    :param padding_length: Padding length embedding layer lstm
    :param epochs:
    :param batch_size:
    :param name: name of model
    :return:
    '''
    X_train, X_test, y_train, y_test = pandas_to_traintestsplit_balanced(data)

    vocab_size = len(set([x for l in data['comment_vect_numeric'].values for x in l]))


    X_train = sequence.pad_sequences(X_train, maxlen=padding_length)
    X_test = sequence.pad_sequences(X_test, maxlen=padding_length)

    network = lstm_model(vocab_size)
    network.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    network.save(name + '.hp5')
    return network, X_test, y_test



def create_submission():
    # data = pd.read_json('./data/processed_test_all_data.json')
    #
    #
    # with open("../data/predication_file.csv", "w") as f:
    #     for i, preds in enumerate(predictions):
    #         f.write("{}, {}, {}, {} , {}, {}, {}\n".formate(id[i], *preds))
    #

    # #
    test_ids = pd.read_csv('./data/test.csv', dtype=np.str)
    print("headers test data", list(test_ids))

    # X_val = np.asarray(data['comment_vect_numeric'])
    # X_val = sequence.pad_sequences(X_val, maxlen=200)


    # network = keras.models.load_model('./my_model.hp5')
    # predictions = network.predict(X_val, verbose=1)

    # submission = pd.DataFrame(predictions)
    # submission.to_csv('./submissions/prediction_file') # 153164


    submission = pd.read_csv('./submissions/prediction_file')
    submission.columns = ['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


    # #
    # for index, row in enumerate(submission):
    #     id = test_ids['id'][index]
    #     predictions = row[1:]
    #     print(id)
    #     print(predictions)

    # print( 'data type ', test_ids['id'].values.dtype)
    #
    #
    submission['id'] = test_ids['id'] #.apply(lambda x: x.strip())
    # submission.insert(0,'id',test_ids['id'])
    print(submission['id'][942])
    print(list(submission))
    print(submission.head)

    submission.to_csv('./predictions_new.csv', columns=['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])


def main():
    data = pd.read_json('./data/processed_train_all_data.json')

<<<<<<< HEAD
    network, X_test, y_test = train_lstm(data,epochs=1)
=======
    network, X_test, y_test = train_lstm(data, epochs=10)
>>>>>>> 9a1674bcac7464138ea4d4b18657224552758d2d

    predictions = network.predict(X_test)
    score, accuracy = network.evaluate(X_test, y_test)

    print('Test score:', score)
    print('Test accuracy:', accuracy)
    print("average auc score", score_function(y_test, predictions))

if __name__ == "__main__":
    main()
    # create_submission()
    # data = pd.read_json('./data/processed_train_1000_data.json')
    # schijt(data)
    # X_train, X_test, y_train, y_test = pandas_to_traintestsplit_balanced(data)