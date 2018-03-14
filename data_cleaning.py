from collections import Counter

import pandas as pd
# Text libs
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

def get_embedding_ids(data, top_perc, replacement_word):
    """
    Get the top k percent of words for each class and represent those with a number
    Other words get the <UNK> token
    :param data:
    :param top_perc:
    :param replacement_word:
    :return:
    """

    embedding_ids = []
    for cat in tqdm(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
        subset = data[data[cat] == 1]
        cnt_words = Counter(([x for l in subset['comment_vect_filtered'].values for x in l]))
        most_common_words = cnt_words.most_common(int(len(cnt_words.keys()) * top_perc))
        embedding_ids.extend([x[0] for x in most_common_words])

    # Convert to set to get unique values, then back to list to support indexing.
    embedding_ids = list(set(embedding_ids))

    # Replace the remaining words in the sentences with '<UNK>'
    embedding_ids.append(replacement_word)

    # Replace the words in the sentences with a numeric value.
    embedding_ids_numeric_dict = {embedding_ids[x]: x for x in
                                  range(len(embedding_ids))}  # Create dictionary from unique words

    return embedding_ids, embedding_ids_numeric_dict


def main(size_to_load=100, top_perc=0.2):
    """
    Load size_to_load of the dataset,
    :param size_to_load:
    :param top_perc:
    :return:
    """
    for value in ['train', 'test']:
        # Load data
        data = pd.read_csv('./data/{}.csv'.format(value), encoding='utf-8', nrows=size_to_load)

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



        # If training, get embedding_ids
        if value == 'train':
            # For each category select the top % words as embedding_ids.
            replacement_word = '<UNK>'
            embedding_ids, embedding_ids_numeric_dict = get_embedding_ids(data, top_perc, replacement_word)

            # Save embedding_ids as csv file.
            df_embedding_ids = pd.DataFrame(list(embedding_ids_numeric_dict.items()), columns=['feature', 'numeric'])
            df_embedding_ids.to_json('./data/embedding_ids.json')

        # Filter comments, then process to numeric values.
        data['comment_vect_filtered'] = tqdm(data['comment_vect_filtered'].apply(lambda x:
                                                                            [replacement_word if i not in embedding_ids else i
                                                                             for i in x]))
        # replace words with numeric values.
        data['comment_vect_numeric'] = tqdm(data['comment_vect_filtered'].apply((lambda x:
                                                                           [embedding_ids_numeric_dict[i]
                                                                            for i in x])))


        data.to_json('./data/processed_{}_data.json'.format(value))
        print("Finished")


if __name__ == "__main__":
    main()