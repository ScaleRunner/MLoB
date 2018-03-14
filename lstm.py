import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed, LSTM
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import pickle
import sklearn

# Load a pickled object
def load_obj(name):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Save a pickled object
def save_obj(obj, name):
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
    # y_predict[y_predict > threshold] = 1
    # y_predict[y_predict < threshold] = 0

    score = []
    for i in range(4):
        print(i)
        score.append(sklearn.metrics.roc_auc_score(y_true[:, i], y_predict[:, i]))

    return np.mean(score)

def main():
    # data = load_obj('processed_train_data')
    data = pd.read_json('./data/processed_train_data.json')


    vocab_size = len(set([x for l in data['comment_vect_numeric'].values for x in l]))

    X_train, X_test, y_train, y_test = pandas_to_traintestsplit(data)

    print(X_test.shape)

    X_train = sequence.pad_sequences(X_train, maxlen=200)
    X_test = sequence.pad_sequences(X_test, maxlen=200)

    network = lstm_model(vocab_size)
    network.fit(X_train, y_train, epochs=1, batch_size=32, verbose=2)
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

if __name__ == "__main__":
    main()
