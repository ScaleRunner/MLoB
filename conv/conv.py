from keras.models import Sequential, optimizers
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Reshape
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import h5py


def load_data(path, test_split=.3):
    df = pd.read_json(path, encoding='utf-8')
    df.dropna()
    X = np.asarray(df['comment_vect_numeric'])
    y = np.asarray(df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

    return X_train, X_test, y_train, y_test


def create_1D_conv(output_size):
    model = Sequential()
    # model.add(Dense(150, activation='relu', input_shape=(None, 1)))
    model.add(Conv1D(200, 3, activation='relu', input_shape=(None, 1)))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D())
    # model.add(Conv1D(200, 3, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D())
    # model.add(Reshape((1, 1)))
    model.add(Dense(output_size, activation='softmax'))

    optimizer = optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    return model


def main():
    print('Loading training data')
    X_train, X_test, y_train, y_test = load_data('../data/processed_train_1000_data.json')
    len_longest_sentence = max([len(x) for x in X_train])
    X_train = np.expand_dims(sequence.pad_sequences(X_train, maxlen=len_longest_sentence), axis=2)
    print(X_train)
    print(X_train.shape)
    print(y_train.shape)
    # print(X_train[0])
    # X_train = np.expand_dims(X_train, axis=-1)

    print('Creating Model')
    network = create_1D_conv(6)

    print('Training network')
    network.fit(X_train, y_train, epochs=2, batch_size=128)
    # score, accuracy = network.evaluate(X_test, y_test)
    # print('loss: {} \n accuracy: {}'.format(score, accuracy))
    #
    # print('Saving model weights')
    # json_model = network.to_json()
    # network.save_weights('conv1d.h5')
    # print('Saved model weights to conv1d.h5')


if __name__ == "__main__":
    main()
