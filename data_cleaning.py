import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
#from gensim.models import word2vec
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

def get_features(data, top_perc, replacement_word):
    features = []
    for cat in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        subset = data[data[cat] == 1]
        cnt_words = Counter([x for l in subset['comment_vect_filtered'].values for x in l])
        most_common_words = cnt_words.most_common(int(len(cnt_words.keys()) * top_perc))
        features.extend([x[0] for x in most_common_words])

    # Convert to set to get unique values, then back to list to support indexing.
    features = list(set(features))

    # Replace the remaining words in the sentences with '<UNK>'
    features.append(replacement_word)

    # Replace the words in the sentences with a numeric value.
    features_numeric_dict = {features[x]: x for x in range(len(features))}  # Create dictionary from unique words


    return features, features_numeric_dict


def load_data_vectorize(size_to_load=1000):

    for value in ['train', 'test']:

         # Load data
        data = pd.read_csv('./data/train.csv', encoding='utf-8').head(size)

        # step 1: Remove NA values.
        data = data.dropna()

        print("column titles", list(data))

        # TODO: Remove stopwords (custom list)
        # cust_stopwords = ['the', 'a', 'an', ',']

        # Step 2: words to list
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        data['comment_vect'] = data['comment_text'].apply(lambda x: tokenizer.tokenize(x))

        ''''
        LSTM
        '''

        # Filter out stop words
        stop = stopwords.words('english')
        data['comment_vect_filtered'] = data['comment_vect'].apply(lambda x: [i for i in x if i not in stop])

        # If training, get features
        if value == 'train':
            # For each category select the top 20% words as features.
            top_perc = 1.0
            replacement_word = '<UNK>'
            features, features_numeric_dict = get_features(data, top_perc, replacement_word)

            # Save features as csv file.
            df_features = pd.DataFrame(list(features_numeric_dict.items()), columns=['feature', 'numeric'])
            df_features.to_csv('./features.csv', encoding='utf-8')

        # Filter comments, then process to numeric values.
        data['comment_vect_filtered'] = data['comment_vect_filtered'].apply(lambda x:
                                                                            [replacement_word if i not in features else i
                                                                                for i in x])

        data['comment_vect_numeric'] = data['comment_vect_filtered'].apply(lambda x:
                                                                           [features_numeric_dict[i]
                                                                                for i in x]) # replace words with numeric values.




        # Save data
        data.to_csv('./data/processed_{}_data.csv'.format(value), encoding='utf-8')


def pandas_to_traintestsplit(dataframe, test_split = 0.0):

    X = np.asarray(dataframe['comment_vect_numeric'])
    y = np.asarray(dataframe[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split, random_state = 42)

    return X_train, X_test, y_train, y_test



def LSTM():
    model = Sequential()

    # Vocabulary = length total unique dict
    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation('softmax'))

    return model


def main():
    # data = load_data_vectorize()
    data = pd.read_csv('/home/winston/PycharmProjects/MLoB/data/processed_train_data.csv')
    print(list(data))


if __name__ == "__main__":
    main()