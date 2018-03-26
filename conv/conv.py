import sklearn
from keras.models import Sequential, optimizers
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Reshape, Embedding
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import keras.backend as K
import numpy as np
from keras.preprocessing import sequence


def load_data(df, test_split=.3):
    df.dropna()
    X = np.asarray(df['comment_vect_numeric'])
    y = np.asarray(df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

    return X_train, X_test, y_train, y_test


def create_1D_conv(longest_sentence, output_size):
    model = Sequential()
    model.add(Embedding(input_dim=longest_sentence, output_dim=2))
    model.add(Conv1D(512, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Conv1D(512, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Conv1D(128, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Conv1D(32, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Conv1D(32, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Conv1D(6, 1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Reshape((output_size,)))
    model.add(Dense(output_size, activation='softmax'))

    optimizer = optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    return model


'''
Column-wise AUC loss function
with 'naive' threshold at .5
'''


# TODO: Is this really column wise ??
# see:  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation
def score_function(y_true, y_predict, threshold=.5):
    score = []
    for i in range(0, len(y_true)):
        label = y_true[i]
        n_correct = 0
        for j in range(0, len(label)):
            predicted = y_predict[i]
            predicted[predicted >= threshold] = 1
            predicted[predicted < threshold] = 0
            if label[j] == predicted[j]:
                n_correct += 1
        score.append(n_correct/len(label))

    return np.mean(score)


def main():
    print('Loading training data')
    df = pd.read_json('../data/processed_train_1000_data.json', encoding='utf-8')
    X_train, X_test, y_train, y_test = load_data(df)
    vocab_size = len(set([x for l in df['comment_vect_numeric'].values for x in l]))
    comment_vecs = np.concatenate((X_train, X_test), axis=0)
    len_longest_sentence = max([len(x) for x in comment_vecs])
    X_train = sequence.pad_sequences(X_train, maxlen=len_longest_sentence)
    X_test = sequence.pad_sequences(X_test, maxlen=len_longest_sentence)

    print('Creating Model')
    network = create_1D_conv(vocab_size, 6)

    print('Training network')
    #  checkpoint
    filepath = "conv1d.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    network.fit(X_train, y_train, epochs=25, batch_size=128, callbacks=callbacks_list)
    score, accuracy = network.evaluate(X_test, y_test)
    print('loss: {} \n accuracy: {}'.format(score, accuracy))

    predictions = network.predict(X_test)
    # print(predictions)
    print("average column wise accuracy", score_function(y_test, predictions, np.mean(predictions)))
    #
    # print('Saving model weights')
    # json_model = network.to_json()
    # network.save_weights('conv1d.h5')
    # print('Saved model weights to conv1d.h5')


if __name__ == "__main__":
    main()
