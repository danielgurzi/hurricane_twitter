from html import unescape

import pandas as pd
import redditcleaner
import regex as re
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

custom_words = list(set(
    list(ENGLISH_STOP_WORDS) + list(stopwords.words('english')) +
    ['california', 'en', 'amp', 'instagram', 'thomas', 'com', 'county', 'org',
     'www', 'https', 'http', 'rt']))

nlp = spacy.load('en',disable=['ner', 'parser'])


def multiply_dataset_retweets(df):
    # take each tweet and multiply the row by the number of retweets as a weight for that tweet
    for ind, row in df.iterrows():
        for rt in range(0, df['retweets'].iloc[ind]):
            df = df.append(pd.Series(row, index=dataframe.columns))


# features = ['username', 'text']

def read_file(file):
    df = pd.read_csv(file)
    #df.drop("Unnamed: 0", axis=1, inplace=True)
    return df


def process_data_by_user(file, disaster_specific_stopwords):
    custom_stopwords = custom_words + disaster_specific_stopwords
    df = read_file(file)
    # convert to lowercase and remove special chars
    remove_extraneous(df, custom_stopwords)
    df['cleaned_tweets'] = clean_tweets(df['text'])
    df['lemmatized_tweets'] = lemmatize_tweets_spacy(df['cleaned_tweets'])

    names = df['username'].unique()
    user_tweet_dict = {}

    for name in names:
        user_tweets = df.loc[df['username'] == name, 'lemmatized_tweets'].tolist()
        user_tweets = [' '.join(utweet) for utweet in user_tweets][0]
        if name in user_tweet_dict.keys():
            user_tweet_dict[name] += ' ' + user_tweets
        else:
            user_tweet_dict[name] = user_tweets

    user_tweet_df = pd.DataFrame.from_dict(user_tweet_dict, orient='index', columns=['text'])
    return user_tweet_df


def remove_extraneous(df, custom_stopwords):
    df['text'] = df['text'].str.lower().apply(lambda x: re.sub("[^a-z\s]", " ", x))
    # remove custom stopwords
    df['text'] = df['text'].apply(
        lambda x: " ".join(word for word in x.split() if word not in custom_stopwords))


def process_data_by_tweet(file, disaster_specific_stopwords):
    custom_stopwords = custom_words + disaster_specific_stopwords
    df = read_file(file)
    # convert to lowercase
    remove_extraneous(df,custom_stopwords)
    df['cleaned_tweets'] = clean_tweets(df['text'])
    df['lemmatized_tweets'] = lemmatize_tweets_spacy(df['cleaned_tweets'])
    return df


def clean_tweets(tweet_series):
    cleaned_tweets = []
    for tweet in tweet_series:
        raw_post = redditcleaner.clean(tweet)
        post_text = BeautifulSoup(unescape(raw_post), 'lxml').get_text()
        alpha_characters_only = re.sub("[^a-zA-Z]+", " ", post_text)
        cleaned_tweets.append(alpha_characters_only)
    return cleaned_tweets


def lemmatize_tweets_spacy(tweet_series):
    lemmas = []
    tokens = []
    for tweet in tweet_series:
        tweet_lemmas, tweet_tokens = lemmatize_tweet_spacy(tweet)
        lemmas.append(' '.join(list(tweet_lemmas.values())))
        tokens.append(tokens)
    return pd.Series(lemmas)


def lemmatize_tweet_spacy(cleaned_tweet):
    # print(f'using spacy on {cleaned_post}')
    doc = nlp(cleaned_tweet)
    tokens_lem = [token.lemma_ for token in doc if 'PRON' not in token.lemma_]
    cleaned_post_list = [tl for tl in tokens_lem]
    save_the_lemmas = {t: w for t, w in zip(tokens_lem, cleaned_post_list)}
    return save_the_lemmas, tokens_lem


def process_data(filename, disaster_specific_stopwords):
    process_data_by_tweet(filename, disaster_specific_stopwords)


if __name__ == '__main__':
    disaster_specific = ['hurricaneharvey', 'harvey', 'houston', 'texas']

    dataframe = process_data_by_tweet('../data/hurricaneharvey/twitter_retrieval/hurricaneharvey_10000.csv', disaster_specific)
    # dataframe = process_data_by_user('../data/mendocinocomplex_pre.csv')
    dataframe.to_csv('../data/hurricaneharvey/processed_datasets/hurricaneharvey_10000.csv')
    print(dataframe)
