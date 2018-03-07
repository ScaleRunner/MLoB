import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import nltk
import string
from tqdm import tqdm, tqdm_pandas
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))

def split_data(features, labels, test_percentage=0.1):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, shuffle=True, test_size=test_percentage)

    return np.array(features_train), np.array(features_test), np.array(labels_train), np.array(labels_test)


def extract_features_from_string(line):
    global stop_words

    features = []

    bag_of_words = [x for x in wordpunct_tokenize(line)]
    sentences = sent_tokenize(line)
    words_per_sentence = np.array([len(wordpunct_tokenize(s)) for s in sentences])
    text_length = float(len(line))
    n_words = float(len(bag_of_words))
    unique_words = [x for x in bag_of_words if x.lower() not in stop_words]
    POS_tags = nltk.pos_tag(line.split())
    tag_freq = nltk.FreqDist(tag for (word, tag) in POS_tags)

    # word_freq = nltk.FreqDist(bag_of_words)

    # Total Number of Characters
    features.append(text_length)

    # Total number of words
    features.append(n_words)

    # Total Number of Sentences
    features.append(len(sentences))

    # Total Number of Punctuations
    punctuation_occurrence = 0
    for punc in string.punctuation:
        punctuation_occurrence += line.count(punc)
    features.append(punctuation_occurrence)

    # Total Number of Parentheses
    paranthesis_occurrences = line.count('(')
    features.append(paranthesis_occurrences)

    # Frequency of questions
    features.append(line.count("?") / len(sentences))

    # Ratio of Characters
    for letter in string.ascii_letters:
        features.append(line.count(letter) / text_length)
        # features.append(line.count(letter))

    # Frequencies of Digits
    for digit in string.digits:
        # features.append(text.count(digit))
        features.append(line.count(digit) / text_length)

    # Function word occurrences:
    for function_word in stop_words:
        features.append(line.count(function_word))

    # Number of unique words, excluded the stopwords
    features.append(len(unique_words))

    # Tag Words Ratio
    important_tags = ["NN", "NNP", "JJ", "DT", "VBP", "VBZ"]
    for tag in important_tags:
        features.append(tag_freq[tag] / n_words)

    # Average Word length
    features.append(np.mean([len(word) for word in unique_words]))

    # Average sentence length
    features.append(words_per_sentence.mean())

    # Sentence length variation
    features.append(words_per_sentence.std())

    return features


def extract_features(df):
    features = []

    texts = df['comment_text']
    for text in texts:
        features.append(extract_features_from_string(text))

    return features


def extract_labels(df):
    categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    labels = []
    for _, sample in tqdm(df.iterrows(), desc="Processing Data", unit="samples", total=len(df)):
        label_series = sample[categories]
        labels.append(label_series.values)
    return labels
