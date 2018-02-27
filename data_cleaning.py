import pandas as pd

# Text libs
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
#from gensim.models import word2vec

import language_check

# Load data
train_data = pd.read_csv('./data/train.csv', encoding='utf-8')
test_data = pd.read_csv('./data/test.csv', encoding='utf-8')

# step 1: fill NA values
#train_data['comment_text'] = train_data['comment_text'].fillna(value='na', inplace=True)
#test_data['comment_text'] = test_data['comment_text'].fillna(value='na', inplace=True)


# Step 2: words to list
# Might be interesting to test with and without stopwords
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
train_data['comment_vect'] = train_data['comment_text'].apply(lambda x: tokenizer.tokenize(x))
#test_data['comment_vect'] = test_data['comment_text'].apply(lambda x: tokenizer.tokenize(x))


def text_to_num(x):
    return [unique_words_dict[i] for i in x]

# Step 3: To numeric values
unique_words = list(set(x for l in train_data['comment_vect'].values.tolist() for x in l)) # All unique words
unique_words_dict = {unique_words[x]:x for x in range(len(unique_words))} # Create dictionairy from unique words
train_data['comment_vect_numeric'] = train_data['comment_vect'].apply(lambda x: text_to_num(x)) # replace words with numeric values.
print(train_data['comment_vect_numeric'])