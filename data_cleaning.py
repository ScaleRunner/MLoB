import pandas as pd
import numpy as np

# Text libs
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
#from gensim.models import word2vec
from collections import Counter

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
    # Replace the words in the sentences with a numeric value.
    features.append(replacement_word)

    features_numeric_dict = {features[x]: x for x in range(len(features))}  # Create dictionairy from unique words

    return features, features_numeric_dict

for value in ['train', 'test']:

    # Load data
    data = pd.read_csv('./data/train.csv', encoding='utf-8').head(100)

    # step 1: Remove NA values.
    data = data.dropna()

    # TODO: Remove stopwords (custom list)
    cust_stopwords = ['the', 'a', 'an', ',']

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
        top_perc = .2
        replacement_word = '<UNK>'
        features, features_numeric_dict = get_features(data, top_perc, replacement_word)

        # Save features as csv file.
        df_features = pd.from_dict(features_numeric_dict)
        df_features.to_csv('./features.csv', encoding='utf-8')

    # Filter comments, then process to numeric values.
    data['comment_vect_filtered'] = data['comment_vect_filtered'].apply(lambda x:
                                                                        [replacement_word if i not in features else i
                                                                            for i in x])

    data['comment_vect_numeric'] = data['comment_vect_filtered'].apply(lambda x:
                                                                       [features_numeric_dict[i]
                                                                        for i in x]) # replace words with numeric values.


    # Save data
    data.to_csv('./processed_{}_data.csv'.format(value), encoding='utf-8')
