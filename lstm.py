import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, TimeDistributed, LSTM
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import sklearn
from sklearn.utils import class_weight
import h5py
import keras

'''
# Load a pickled object
def load_obj(name):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Save a pickled object
def save_obj(obj, name):
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
'''

def pandas_to_traintestsplit(dataframe, test_split=.3):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for kind in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        data = dataframe[dataframe[kind] == 1]
        x = np.asarray(data['comment_vect_numeric'])
        y = np.asarray(data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(x, y, test_size=5)
        X_train.append(X_train_new)
        X_test.append(X_test_new)
        y_train.append(y_train_new)
        y_test.append(y_test_new)


    data = dataframe[(dataframe['toxic'] == 0) & (dataframe['severe_toxic'] == 0) & (dataframe['obscene'] == 0)
                & (dataframe['threat'] == 0) & (dataframe['insult'] == 0) & (dataframe['identity_hate'] == 0)]
    x = np.asarray(data['comment_vect_numeric'])
    y = np.asarray(data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(x, y, test_size=5)
    X_train.append(X_train_new)
    X_test.append(X_test_new)
    y_train.append(y_train_new)
    y_test.append(y_test_new)

    
    return np.asarray(X_train).flatten(), np.asarray(X_test).flatten(), np.asarray(y_train).flatten(), np.asarray(y_test).flatten()


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

    score = sklearn.metrics.roc_auc_score(y_true, np.exp(y_predict))

    print("score = ", score)
    return np.mean(score)


def train_lstm(data, padding_length=200,epochs=5, batch_size=64,name='my_model'):
    X_train, X_test, y_train, y_test = pandas_to_traintestsplit(data)

    vocab_size = len(set([x for l in data['comment_vect_numeric'].values for x in l]))


    X_train = sequence.pad_sequences(X_train, maxlen=padding_length)
    X_test = sequence.pad_sequences(X_test, maxlen=padding_length)

    network = lstm_model(vocab_size)
    network.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=1)

    network.save(name + '.hp5')
    return network, X_test, y_test



def create_submission():
    data = pd.read_json('./data/processed_test_all_data.json')

    X_val = np.asarray(data['comment_vect_numeric'])
    X_val = sequence.pad_sequences(X_val, maxlen=200)


    network = keras.models.load_model('./my_model.hp5')
    predictions = network.predict(X_val)


    submission = pd.DataFrame(predictions)
    submission.columns = ['id','toxic','severe_toxic', 'obscene','threat','insult','identity_hate']

    ids = data['id']
    submission['id'] = ids

    submission.to_csv("predictions_kaggle_test.csv")


def main():
    # data = load_data_vectorize()
    data = pd.read_json('./data/processed_train_all_data.json')

    network, X_test, y_test = train_lstm(data,epochs=1)


    predictions = network.predict(X_test)
    score, accuracy = network.evaluate(X_test, y_test)

    print('Test score:', score)
    print('Test accuracy:', accuracy)

    print("average auc score", score_function(y_test, predictions))

if __name__ == "__main__":
    # main()
    data = pd.read_json('./data/processed_train_all_data.json')

    X_train, X_test, y_train, y_test = pandas_to_traintestsplit(data)