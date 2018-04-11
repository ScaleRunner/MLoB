from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import nltk
import pickle
from tqdm import tqdm, tqdm_pandas
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm, tqdm_pandas
tqdm.pandas(tqdm())
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, optimizers
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class

desired_width = 4000
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
stop = stopwords.words('english')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Save a pickled object
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# Load Google's pre-trained Word2Vec model.
# tweet_w2v = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
#
# save_obj(tweet_w2v, 'tweet_w2v')
#
#
# print("Build a Tf-Id matrix")
#
# dataframe = pd.read_json('./comment_tokenized_and_lower.json')
#
# X = np.asarray(dataframe['comment_tokenized_lower'])
#
# print(X.shape)
#
# vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=2)
# matrix = vectorizer.fit_transform([x for x in X])
# tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
# print('vocab size :', len(tfidf))
# save_obj(tfidf, 'tfidf')

tfidf = load_obj('tfidf')
tweet_w2v = load_obj('tweet_w2v')


def load_data(path, size=None):
    data = pd.read_csv(path, encoding='utf-8', nrows=size)
    data = data.dropna()

    # Step 2: words to list
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    data['comment_tokenized'] = data['comment_text'].progress_apply(lambda x: tokenizer.tokenize(x))

    data['comment_tokenized_lower'] = data['comment_tokenized'].progress_apply(lambda x: [i.lower() for i in x if i not in stop])

    data.to_json('./comment_tokenized_and_lower.json')

    return data



def tweet_tfidf__w2vec_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:
            # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

def extract_features(df):

  df = pd.DataFrame(df)

  features = pd.DataFrame()
  temporary = pd.DataFrame()

  features['total_length'] = df['comment_tokenized'].apply(len)
  features['capitals'] = df['comment_tokenized'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
  features['caps_vs_length'] = features.apply(lambda row: float(row['capitals']) / float(row['total_length']),
                                  axis=1)
  features['num_exclamation_marks'] = df['comment_tokenized'].apply(lambda comment: comment.count('!'))
  features['num_question_marks'] = df['comment_tokenized'].apply(lambda comment: comment.count('?'))
  features['num_double_quotation'] = df['comment_tokenized'].apply(lambda comment: comment.count('"'))

  features['num_punctuation'] = df['comment_tokenized'].apply(
      lambda comment: sum(comment.count(w) for w in string.punctuation))
  features['num_symbols'] = df['comment_tokenized'].apply(
      lambda comment: sum(comment.count(w) for w in '*&$%#@'))
  features['num_words'] = df['comment_tokenized'].apply(lambda comment: len(comment))
  features['num_unique_words'] = df['comment_tokenized_lower'].apply(
      lambda comment: len(set(w for w in comment)))
  features['words_vs_unique'] = features['num_unique_words'] / features['num_words']


  # List of swear word
  swear_words = pd.read_csv('../swear_word.txt', header=None)
  swear_words.drop_duplicates(inplace=True)

  # print(swear_words)
  for swear_word in tqdm(swear_words[0]):
      features[swear_word] = df['comment_tokenized_lower'].apply(lambda comment: comment.count(swear_word))


  # Tf-IDF averaged word2vec vector
  y = df['comment_tokenized_lower'].progress_apply(lambda comment: tweet_tfidf__w2vec_vector(comment, 300)).values


  # Extract column per each word2vec dimension
  for id in range(200):
      features[str(id)] = [dim[id] for vec in y for dim in vec]


  print(list(features))


  # Remove feature columns with all zero -> no information
  features = features[features.columns[features.max() > 0]]


  numpy_features = features.as_matrix()
  np.save('./numpy_features_not_normalized',numpy_features)


  scaler = MinMaxScaler()
  norm_features = scaler.fit_transform(numpy_features)

  np.save('./numpy_normfeatures',norm_features)

  return norm_features

def create_model(n_features, output_size):
  model = Sequential([
      Dense(256, activation='relu', input_shape=(n_features,)),
      Dense(128, activation='relu'),
      Dense(128, activation='relu'),
      Dense(256, activation='relu'),

      Dense(output_size, activation='softmax')
  ])

  # Multiclass uses categorical cross-entropy
  optimizer = optimizers.SGD(lr=1e-6)
  model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
  print(model.summary())
  return model

def score_function(y_true, y_predict):
  """
  :param y_true:
  :param y_predict:
  :return: Mean averaged column wise AUC score
  """
  scores = []
  for i in range(6):
      scores.append(sklearn.metrics.roc_auc_score(y_true[:, i], y_predict[:, i]))
  print("score = ", np.mean(scores))
  return np.mean(scores)


def batch_generator(X,y, batch_size=128):
  # Create empty arrays to contain batch of features and labels#

  batch_features = np.zeros((batch_size, 64, 64, 3))
  batch_labels = np.zeros((batch_size, 1))

  while True:
      for i in range(batch_size):
          # choose random index in features
          index = random.choice(len(features), 1)
          batch_features[i] = some_processing(features[index])
          batch_labels[i] = labels[index]
      yield batch_features, batch_labels

  return

def main():

  # dataframe  = load_data('../../data/train.csv')

  dataframe = pd.read_json('./comment_tokenized_and_lower.json')


  X = np.load('./numpy_features_not_normalized.npy')
  # X = extract_features(dataframe)
  y = np.asarray(dataframe[['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']])

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

  network = create_model(X.shape[1], 6)

  file_path = "./weights_feature_extractor"
  checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
  callbacks_list = [checkpoint, early]  # early

  network.fit(X_train, y_train,
            epochs=25,
            batch_size=128,
            verbose=1,
            validation_data=(X_test, y_test),
            shuffle=True,
            callbacks=callbacks_list)



  y_pred = network.predict(X_test, verbose=1)
  print(score_function(y_test,y_pred))


  # network.load_weights(file_path)

  # test = pd.read_csv("../../data/test.csv")
  #
  # y_test = network.predict(X_test, verbose=1)
  #
  # sample_submission = pd.read_csv("../../data/sample_submission.csv")
  #
  # sample_submission[list_classes] = y_test
  #
  # sample_submission.to_csv("feature_extractor.csv", index=False)


if __name__ == "__main__":
  main()
