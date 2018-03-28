import glob
import pandas as pd
pd.options.mode.chained_assignment = None
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from sklearn.preprocessing import scale
import pickle
from tweet_tokenizer import CustomTokenizer
n=1000000; n_dim = 200
tweet_tokenizer = CustomTokenizer()
from keras.utils import to_categorical
dimension_w2v = 200
from tensorflow.python.client import device_lib
from sklearn.model_selection import KFold
import tensorflow
sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from keras import metrics
# fix random seed for reproducibility
seed = 7
dimension_w2v = 200
from keras import models
# model = models.load_model(model_path, custom_objects= {'f1_score': f1_score})


def tokenize(tweet):
    try:
        tweet = (tweet.lower())
        tokens = tokenizer.tokenize(tweet)
        # Remove tokens starting with @ and #
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'NC'

#  http://help.sentiment140.com/for-students/
# Massive Positive /negative dataset ~ 1 600 000  long
def getDatData():
    df = pd.read_csv('/home/henkdetank/PycharmProjects/TextMining/Data/training.1600000.processed.noemoticon.csv', sep=',', error_bad_lines=False,encoding='latin-1')
    df.drop(df.columns[[1, 2, 3, 4]],axis=1,inplace=True)
    df.columns = ['sentiment', 'tweet']
    df['sentiment'] = df['sentiment'].map({4: 1, 0: -1})

    print('DatData met duplicates ', df.shape)
    df.drop_duplicates('tweet')
    print('DatData zonder duplicates ', df.shape)

    return df['sentiment'], df['tweet']


#  All SemEvalData 2013 - 2016  ~ 52 000 long
def getAllSemTaskAData():
    dfEmpty = pd.DataFrame()
    path = "/home/henkdetank/PycharmProjects/TextMining/TaskATextMining/*"
    files = glob.glob(path)
    frames = []
    for name in files:
        dtemp = pd.read_csv(name, sep='\t', error_bad_lines=False)
        dtemp.columns = ['id', 'sentiment', 'tweet']
        frames.append(dtemp)

    df = pd.concat(frames)

    df['sentiment'] = df['sentiment'].replace(['objective'], 'neutral')
    print(df.groupby('sentiment').count())

    df['sentiment'] = df['sentiment'].replace(['objective'], 0)
    df['sentiment'] = df['sentiment'].replace(['neutral'], 0)
    df['sentiment'] = df['sentiment'].replace(['positive'], 1)
    df['sentiment'] = df['sentiment'].replace(['negative'], -1)


    df.drop(['id'], axis=1, inplace=True)

    filtered_df = df[df.isnull()]

    print(df.shape)
    df.drop([11063], inplace=True)
    print(df.shape)


    return df['sentiment'], df['tweet']


def load_obj(name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

def save_obj(obj, name):
        with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def train_w2v():

    y_semeval, x_semeval = getAllSemTaskAData()
    y_emoticon, x_emoticon = getDatData()

    X = pd.concat([x_semeval, x_emoticon])
    X = pd.concat([x_semeval, x_emoticon])
    print("FUll x shape met duplicates", X.shape)
    X = X.drop_duplicates()
    print("Full x zonder duplicates", X.shape)

    y = pd.concat([y_semeval, y_emoticon])
    X = (tqdm([tokenize(tweet) for tweet in X]))

    tweet_w2v = Word2Vec(size=dimension_w2v, min_count=1)
    tweet_w2v.build_vocab([word for word in X])
    tweet_w2v.train(tqdm([word for word in X]), total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

    print("Build a Tf-Id matrix")
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=5)
    matrix = vectorizer.fit_transform([x for x in X])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))

    save_obj(tweet_w2v, "tweet_w2v")
    save_obj(tfidf, "tfidf")


def return_full_dataset():
    # y_semeval, x_semeval = getAllSemTaskAData()
    # y_emoticon, x_emoticon = getDatData()
    #
    # X = pd.concat([x_semeval, x_emoticon])
    # print("FUll x shape met duplicates", X.shape)
    # full_X = X.drop_duplicates()
    #
    # # print(Y.head())
    # # Y['sentiment'] = Y['sentiment'].replace(['objective'], 'neutral')
    #
    # print("SemEvel counter ", Counter(y_semeval))
    # print("SemEvel Y emoticaon", Counter(y_emoticon))
    #
    # Y = pd.concat([y_semeval, y_emoticon])
    # print("Tokenizing dataset")
    # X = (([tokenize(tweet) for tweet in X]))
    # save_obj(X, "xdatatokenized")
    # save_obj(Y, "ydata")
    # print("Retrieved Full Dataset")
    X = load_obj("xdatatokenized")
    Y = load_obj("ydata")
    return X, Y

def vectors(X_train, X_test):

    tweet_w2v = load_obj("tweet_w2v")
    tfidf = load_obj("tfidf")
    dimension_w2v =200

    def tweet_tfidf__w2vec_vector(tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tokens:
            try:
                vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
                count += 1.
            except KeyError:
                # Token not in corpus
                continue
        if count != 0:
            vec /= count
        return vec


    train_vecs_w2v = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimension_w2v) for z in tqdm(map(lambda x: x, X_train))]))
    train_vecs_w2v = scale(train_vecs_w2v)
    save_obj(train_vecs_w2v,"train_vecs_w2v")


    test_vecs_w2v = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimension_w2v) for z in tqdm(map(lambda x: x, X_test))]))
    test_vecs_w2v = scale(test_vecs_w2v)
    save_obj(test_vecs_w2v,"test_vecs_w2v")
    print("Finished vectorizing train and test tweet")


def vectorize_one_set(X):
    tweet_w2v = load_obj("tweet_w2v")
    tfidf = load_obj("tfidf")
    dimension_w2v = 200

    def tweet_tfidf__w2vec_vector(tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tokens:
            try:
                vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
                count += 1.
            except KeyError:
                # Token not in corpus
                continue
        if count != 0:
            vec /= count
        return vec

    X = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimension_w2v) for z in tqdm(map(lambda x: x, X))]))
    save_obj(X, "fullxvectorized")
    return X

def random_forest():

    # X,Y = return_full_dataset()
    Y,X = getAllSemTaskAData()

    X = load_obj("fullxvectorized")

    X_train_SemEvalA, X_test_SemEvalA, y_train_SemEvalA, y_test_SemEvalA = train_test_split(X, Y, test_size=0.20, random_state=42)

    vectors(X_train_SemEvalA,X_test_SemEvalA)

    train_vecs_w2v = load_obj("train_vecs_w2v")
    test_vecs_w2v = load_obj('test_vecs_w2v')

    clf = RandomForestClassifier()
    clf.fit(train_vecs_w2v, y_train_SemEvalA)
    predictions = (clf.predict(test_vecs_w2v))

#     0.679766002626

def naive_bayes():
    # X,Y = return_full_dataset()
    Y, X = getAllSemTaskAData()
    X = ([tokenize(tweet) for tweet in X])
    print(Y)
    X_train_SemEvalA, X_test_SemEvalA, y_train_SemEvalA, y_test_SemEvalA = train_test_split(X, Y, test_size=0.20,
                                                                                            random_state=42)
    print("Wat is dezze")
    vectors(X_train_SemEvalA, X_test_SemEvalA)

    train_vecs_w2v = load_obj("train_vecs_w2v")
    test_vecs_w2v = load_obj('test_vecs_w2v')

    print(" y_train: ", y_train_SemEvalA)
    print("waar gaat dit")
    clf = naive_bayes()
    clf.fit(train_vecs_w2v, y_train_SemEvalA)
    print(" whaddup ")
    predictions = (clf.predict(test_vecs_w2v))
    print(predictions)

    print(accuracy_score(y_test_SemEvalA, predictions))
    print(confusion_matrix(y_test_SemEvalA, predictions))

def k_fold():


    X,Y = return_full_dataset()

    # X = vectorize_one_set(X)
    X = load_obj("fullxvectorized")

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []

    print("Availabe devices to train network: ", device_lib.list_local_devices())

    print(len(X))

    X = np.asarray(X)
    Y = np.asarray(Y)
    cateY =  to_categorical(Y, num_classes=3)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    print(X.shape)
    print(Y.shape)

    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=dimension_w2v))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy',metrics.mae, metrics.categorical_accuracy])

    f1scores = []
    for train, test in (kfold.split(X,Y)):
        print(len(train))
        print("test counter", Counter(Y[test]))
        model.fit(X[train], cateY[train], epochs=5, batch_size=32, verbose=2)


        # Predict F1 Score
        y_predict = model.predict_classes(X[test])
        y_predict[y_predict == 2] = -1
        print(y_predict)
        print(Y[test])

        print("Confusion matrix ", confusion_matrix(Y[test], y_predict))

        f1scores.append(f1_score(y_true=Y[test], y_pred=y_predict, average='macro'))

        # evaluate the model
        scores = model.evaluate(X[test], cateY[test], verbose=0)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(f1scores), np.std(f1scores)))


def main():


    y_semeval, x_semeval = getAllSemTaskAData()
    y_emoticon, x_emoticon = getDatData()

    X = pd.concat([x_semeval, x_emoticon])
    y = pd.concat([y_semeval, y_emoticon])
    X = ([tokenize(tweet) for tweet in X])

    X_train_SemEvalA, X_test_SemEvalA, y_train_SemEvalA, y_test_SemEvalA = train_test_split(x_semeval, y_semeval, test_size=0.20, random_state=42)

    save_obj(X_train_SemEvalA, "X_train")
    save_obj(X_test_SemEvalA, "X_test")
    save_obj(y_train_SemEvalA, "y_train")
    save_obj(y_test_SemEvalA, "y_test")


    tweet_w2v = load_obj("tweet_w2v")
    tfidf = load_obj("tfidf")

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
    '''
    print("waar gaat")

    train_vecs_w2v = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimension_w2v) for z in tqdm(map(lambda x: x, X_train_SemEvalA))]))
    print("dit fout hier")
    train_vecs_w2v = scale(train_vecs_w2v)
    print('of hier')
    save_obj(train_vecs_w2v,"train_vecs_w2v")

    print("niet hier ")


    test_vecs_w2v = np.concatenate(([tweet_tfidf__w2vec_vector(z, dimension_w2v) for z in tqdm(map(lambda x: x, X_test_SemEvalA))]))
    test_vecs_w2v = scale(test_vecs_w2v)
    save_obj(test_vecs_w2v,"test_vecs_w2v")
    

    train_vecs_w2v = load_obj("train_vecs_w2v")
    test_vecs_w2v = load_obj('test_vecs_w2v')

    dimension_w2v = 200
    y_train_SemEvalA = load_obj("y_train")
    y_test_SemEvalA = load_obj("y_test")

    x, y = getDatData()
    X, Y = getAllSemTaskAData()

    y_train_SemEvalA = to_categorical(y_train_SemEvalA, num_classes=3)
    y_test_SemEvalA =  to_categorical(y_test_SemEvalA, num_classes=3)

    print(y_train_SemEvalA.shape)
    print(y_test_SemEvalA.shape)

    print("trololo gpu", device_lib.list_local_devices())

    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=dimension_w2v))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    values = model.fit(train_vecs_w2v, y_train_SemEvalA, epochs=5, batch_size=32, verbose=2)

    print((values))

    score = model.evaluate(test_vecs_w2v, y_test_SemEvalA, batch_size=128, verbose=2)
    print(score[1])
    '''


def fitthamodel():
    X, Y = return_full_dataset()

    #
    # Y,X  = getAllSemTaskAData()
    # X = ([tokenize(tweet) for tweet in X])


    X_train_SemEvalA, X_test_SemEvalA, y_train_SemEvalA, y_test_SemEvalA = train_test_split(X, Y, test_size=0.20,
                                                                                            random_state=42)

    vectors(X_train_SemEvalA, X_test_SemEvalA)

    y_train_SemEvalA = to_categorical(y_train_SemEvalA, num_classes=3)
    y_test_SemEvalA = to_categorical(y_test_SemEvalA, num_classes=3)

    train_vecs_w2v = load_obj("train_vecs_w2v")
    test_vecs_w2v = load_obj('test_vecs_w2v')

    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=dimension_w2v))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_vecs_w2v, y_train_SemEvalA, epochs=5, batch_size=32, verbose=2)

    predictions = model.predict(test_vecs_w2v)
    probability_predictions = model.predict_proba(test_vecs_w2v)
    print("probabiltiy predictions = ", probability_predictions)

    # print("Accuracy = " , accuracy_score(y_test_SemEvalA, predictions,normalize=False))
    print("Confusion matrix ", confusion_matrix(y_test_SemEvalA, predictions))

    score = model.evaluate(test_vecs_w2v, y_test_SemEvalA, batch_size=128, verbose=2)
    print("score accuracy", score[1])



if __name__ == '__main__':
    # train_w2v()
    k_fold()
    # main()
    # fitthamodel()
    # return_full_dataset()
    # random_forest()
    # naive_bayes()
    # return_full_dataset()