################################################################################
#  This script provides a general outline of how you can train a model on GCP  #
#  Authors: Mick van Hulst, Dennis Verheijden                                  #
################################################################################

from __future__ import absolute_import
import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, GlobalMaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing import text, sequence
import sklearn
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint


import json
from tensorflow.python.lib.io import file_io

max_features = 20000
maxlen = 100


def load_data(path):
    """
    Loading the data and turning it into a pandas dataframe
    :param path: Path to datafile; Can be predefined as shown above.
    :return: pandas dataframe
    """

    data = pd.read_csv(path)
    return data



def create_model():
    """
    In here you can define your model
    NOTE: Since we are only saving the model weights, you cannot load model weights that do
    not have the exact same architecture.
    :return:
    """
    embed_size = 512
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(1024, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model


def main(train_file, test_file, job_dir):
    train = load_data(train_file)
    test = load_data(test_file)
    train = train.sample(frac=1)

    model = create_model()
    batch_size = 32
    epochs = 25

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

    file_path="./weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, 	mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early] #early


    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, 	  		callbacks=callbacks_list)

    score, accuracy = model.evaluate(X_validation, y_validation)
    print('Test score:', score)
    print('Test accuracy:', accuracy)



    predictions = model.predict(X_test)
    # TODO: Kaggle competitions accept different submission formats, so saving the predictions is 		up to you
    # Save model weights
    model.save('model.h5')

    # Save model on google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    """
    The argparser can also be extended to take --n-epochs or --batch-size arguments
    """
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      required=True
    )
    parser.add_argument(
      '--test-files',
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
    print('args: {}'.format(arguments))

    main(args.train_files, args.test_files, args.job_dir)
