import pandas as pd
import numpy as np

# Text libs
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
#from gensim.models import word2vec
from collections import Counter
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

def get_embedding_ids(data, top_perc, replacement_word):
    embedding_ids = []
# Save a pickled object
def save_obj(obj, name):
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

'''
Get the top k percent of words for each class and represent those with a number
Other words get the <UNK> token
'''
def get_features(data, top_perc, replacement_word):
    features = []
    for cat in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        subset = data[data[cat] == 1]
        cnt_words = Counter([x for l in subset['comment_vect_filtered'].values for x in l])
        most_common_words = cnt_words.most_common(int(len(cnt_words.keys()) * top_perc))
        embedding_ids.extend([x[0] for x in most_common_words])

    # Convert to set to get unique values, then back to list to support indexing.
    embedding_ids = list(set(embedding_ids))

    # Replace the remaining words in the sentences with '<UNK>'
    embedding_ids.append(replacement_word)

    # Replace the words in the sentences with a numeric value.
    embedding_ids_numeric_dict = {embedding_ids[x]: x for x in range(len(embedding_ids))}  # Create dictionairy from unique words

    return embedding_ids, embedding_ids_numeric_dict

'''
Load size_to_load of the dataset, 
'''
def load_data_vectorize(size_to_load=10000, top_perc = 1.0):

    for value in ['train', 'test']:

         # Load data
        data = pd.read_csv('./data/train.csv', encoding='utf-8').head(size_to_load)

        # step 1: Remove NA values.
        data = data.dropna()

        # TODO: Remove stopwords (custom list)
        # cust_stopwords = ['the', 'a', 'an', ',']

        # Step 2: words to list
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        data['comment_vect'] = data['comment_text'].apply(lambda x: tokenizer.tokenize(x))


    # Filter out stop words and convert the resulting set to lower case strings.
    stop = stopwords.words('english')
    data['comment_vect_filtered'] = data['comment_vect'].apply(lambda x: [i.lower() for i in x if i not in stop])
        # Filter out stop words
        stop = stopwords.words('english')
        data['comment_vect_filtered'] = data['comment_vect'].apply(lambda x: [i for i in x if i not in stop])

    # If training, get embedding_ids
    if value == 'train':
        # For each category select the top 20% words as embedding_ids.
        top_perc = .2
        replacement_word = '<UNK>'
        embedding_ids, embedding_ids_numeric_dict = get_embedding_ids(data, top_perc, replacement_word)
        # If training, get features
        if value == 'train':
            # For each category select the top 20% words as features.
            replacement_word = '<UNK>'
            features, features_numeric_dict = get_features(data, top_perc, replacement_word)

        # Save embedding_ids as csv file.
        df_embedding_ids = pd.DataFrame(list(embedding_ids_numeric_dict.items()), columns=['feature', 'numeric'])
        df_embedding_ids.to_csv('./embedding_ids.csv', encoding='utf-8')
            # Save features as csv file.
            df_features = pd.DataFrame(list(features_numeric_dict.items()), columns=['feature', 'numeric'])
            df_features.to_csv('./features.csv', encoding='utf-8')

    # Filter comments, then process to numeric values.
    data['comment_vect_filtered'] = data['comment_vect_filtered'].apply(lambda x:
                                                                        [replacement_word if i not in embedding_ids else i
                                                                            for i in x])
        # Filter comments, then process to numeric values.
        data['comment_vect_filtered'] = data['comment_vect_filtered'].apply(lambda x:
                                                                            [replacement_word if i not in features else i
                                                                                for i in x])

    data['comment_vect_numeric'] = data['comment_vect_filtered'].apply(lambda x:
                                                                       [embedding_ids_numeric_dict[i]
                                                                        for i in x]) # replace words with numeric values.
        data['comment_vect_numeric'] = data['comment_vect_filtered'].apply(lambda x:
                                                                           [(features_numeric_dict[i])
                                                                                for i in x]) # replace words with numeric values.

        # Save data
        save_obj(data, 'processed_{}_data'.format(value))
        # data.to_csv('./data/processed_{}_data.csv'.format(value), encoding='utf-8')


def pandas_to_traintestsplit(dataframe, test_split = .3):

    X = np.asarray(dataframe['comment_vect_numeric'])
    y = np.asarray(dataframe[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split, random_state = 42)

    return X_train, X_test, y_train, y_test



def lstm_model(vocabulary, hidden_size = 200):


    model = Sequential()
    # Vocabulary = length total unique dict
    # Embedding layer creates word2vec vector
    model.add(Embedding(input_dim=vocabulary ,output_dim=hidden_size))
    model.add(LSTM(hidden_size, dropout=.2))
    model.add((Dense(6,activation='sigmoid')))

    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())


    return model


def main():
    # data = load_data_vectorize()
    data = load_obj('processed_train_data')

    vocab_size  = len(set([x for l in data['comment_vect_numeric'].values for x in l]))

    X_train, X_test, y_train, y_test = pandas_to_traintestsplit(data)


    X_train = sequence.pad_sequences(X_train, maxlen=200)
    X_test = sequence.pad_sequences(X_test, maxlen=200)

    network = lstm_model(vocab_size)
    network.fit(X_train, y_train, nb_epoch=1, batch_size=32, verbose=2)
    score, accuracy = network.evaluate(X_test, y_test)

    # network.save('./data/lstm')


    predications = network.predict(X_test)
    print(predications[0])
    print(predications[20])
    print(predications[40])
    print(predications[22])
    print(predications[25])

    print('Test score:', score)
    print('Test accuracy:', accuracy)

    print("average auc score" , score_function(y_test,predications))


'''
Column-wise AUC loss function
with 'naive' threshold at .5
'''
# TODO: Is this really column wise ??  see:  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation
def score_function(y_true, y_predict,threshold =.5):
    y_predict[y_predict > threshold] = 1
    y_predict[y_predict < threshold] = 0

    score = sklearn.metrics.roc_auc_score(y_true, np.exp(y_predict))

    print("score = ", score)
    return np.mean(score)




if __name__ == "__main__":
    main()
