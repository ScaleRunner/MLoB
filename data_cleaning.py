import pandas as pd

# Text libs
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
#from gensim.models import word2vec
from collections import Counter

import language_check

# Load data
train_data = pd.read_csv('./data/train.csv', encoding='utf-8')
test_data = pd.read_csv('./data/test.csv', encoding='utf-8')

# step 1: Remove NA values.

# Step 2: Remove stopwords (custom list)
stopwords = ['the', 'a', 'an', ',']


# Step 2: words to list
# Might be interesting to test with and without stopwords
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
train_data['comment_vect'] = train_data['comment_text'].apply(lambda x: tokenizer.tokenize(x))
#test_data['comment_vect'] = test_data['comment_text'].apply(lambda x: tokenizer.tokenize(x))

''''
This is for LSTM
'''

# Filter out stop words
stop = stopwords.words('english')
train_data['comment_vect_filtered'] = train_data['comment_vect'].apply(lambda x: [i for i in x if i not in stop])

# Count all words
#unique_words = Counter([x for l in train_data['comment_vect'].values.tolist() for x in l])# All unique words

top_words = []

toxic_set = train_data[train_data['toxic'] == 1]
unique_words_toxic = Counter([x for l in train_data['comment_vect_filtered'].values.tolist() for x in l])



severe_toxic_set = train_data[train_data['severe_toxic'] == 1]
obscene_set = train_data[train_data['obscene'] == 1]
threat_set = train_data[train_data['threat'] == 1]
insult_set = train_data[train_data['insult'] == 1]
identity_hate_set = train_data[train_data['identity_hate'] == 1]




# For each category select the top 20% words.

# Replace the remaining words in the sentences with '<UNK>'

# Replace the words in the sentences with a numeric value.


'''
unique_words = list(set(x for l in train_data['comment_vect'].values.tolist() for x in l)) # All unique words
unique_words_dict = {unique_words[x]:x for x in range(len(unique_words))} # Create dictionairy from unique words
train_data['comment_vect_numeric'] = train_data['comment_vect'].apply(lambda x: text_to_num(x)) # replace words with numeric values.
print(train_data['comment_vect_numeric'])
'''