import sklearn
from keras.models import Sequential, optimizers, load_model
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Reshape, Embedding, Flatten, Convolution1D, GlobalAveragePooling1D
from keras.preprocessing import text
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import keras.backend as K
import numpy as np
from keras.preprocessing import sequence


def train_test_split_balanced(X, y, test_size=0.25):
    label_counts = [[]] * len(y[0])
    print(len(y))
    for i in range(0, len(y)):
        for j in range(0, len(y[0])):
            if y[i][j] == 1:
                label_counts[j].append(y[i])

    for i in range(0, len(label_counts)):
        print(len(label_counts[i]))
    print(np.argmin(label_counts))
    lowest_count_idx = np.argmin(label_counts)
    test_sample_per_label = test_size * lowest_count_idx
    print(test_sample_per_label)
    return None, None, None, None


def load_data(df, test_split=.3):
    df.dropna()
    X = np.asarray(df['comment_vect_numeric'])
    y = np.asarray(df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=None)
    # X_train, X_test, y_train, y_test = train_test_split_balanced(X, y)

    return X_train, X_test, y_train, y_test


def create_1D_conv(longest_sentence, output_size):
    model = Sequential()
    model.add(Embedding(input_dim=longest_sentence, output_dim=2))
    model.add(Conv1D(512, 50, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, 25, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(128, 13, activation='relu'))
    model.add(MaxPooling1D())
    # model.add(Conv1D(512, 5, activation='relu'))
    # model.add(MaxPooling1D())
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    optimizer = optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    return model


def score_function(y_true, y_predict):
    scores = []
    for i in range(len(y_predict[0])):
        scores.append(sklearn.metrics.roc_auc_score(y_true[:, i], y_predict[:, i]))
    print("score = ", np.mean(scores))
    return np.mean(scores)



def main():
    max_features = 20000
    maxlen = 906
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("MLoB").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_train = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("MLoB").values

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    print('Creating Model')
    network = create_1D_conv(20000, 6)

    network.fit(X_train, y_train, epochs=2, batch_size=128)
    print('Saving model weights')
    json_model = network.to_json()
    network.save_weights('conv1d.h5')
    print('Saved model weights to conv1d.h5')

    print('Loading model weights')
    network.load_weights('conv1d.h5')

    print(X_test.shape)
    y_test = network.predict(X_test, batch_size=128)
    print(y_test.shape)

    sample_submission = pd.read_csv("../data/sample_submission.csv")
    print(sample_submission.shape)

    sample_submission[list_classes] = y_test
    print(sample_submission)

    sample_submission.to_csv("conv1d_2_prediction.csv", index=False)


if __name__ == "__main__":
    main()
