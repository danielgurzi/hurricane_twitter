import json
import pickle
from html import unescape
from sense2vec import Sense2VecComponent

import GetOldTweets3 as got
import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import redditcleaner
import regex as re
import spacy
from nltk.cluster import KMeansClusterer
import nltk

import uuid
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_samples, silhouette_score


class Workflow(object):
    """Full workflow"""

    @staticmethod
    def _get_tweets_with_got3(query, start='2006-03-21', end=datetime.date.today().strftime("%Y-%m-%d"),
                              maxtweets=1000):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query) \
            .setSince(start) \
            .setUntil(end) \
            .setMaxTweets(maxtweets)
        tweet = got.manager.TweetManager.getTweets(tweetCriteria)
        # tweet_dict = tweetCriteria.__dict__
        df = pd.DataFrame([t.__dict__ for t in tweet])
        return df

    @staticmethod
    def _remove_extraneous(df, custom_stopwords):
        df['text'] = df['text'].str.lower().apply(lambda x: re.sub("[^a-z\s]", " ", x))
        # remove custom stopwords
        df['text'] = df['text'].apply(
            lambda x: " ".join(word for word in x.split() if word not in custom_stopwords))

    @staticmethod
    def _clean_tweets(self, tweet_series):
        cleaned_tweets = []
        for tweet in tweet_series:
            raw_post = redditcleaner.clean(tweet)
            post_text = BeautifulSoup(unescape(raw_post), 'lxml').get_text()
            alpha_characters_only = re.sub("[^a-zA-Z]+", " ", post_text)
            cleaned_tweets.append(alpha_characters_only)
        return cleaned_tweets

    @staticmethod
    def _multiply_dataset_retweets(df):
        # take each tweet and multiply the row by the number of retweets as a weight for that tweet
        for ind, row in df.iterrows():
            for rt in range(0, df['retweets'].iloc[ind]):
                df = df.append(pd.Series(row, index=df.columns))

    @staticmethod
    def _read_file(file):
        df = pd.read_csv(file)
        df.drop("Unnamed: 0", axis=1, inplace=True)
        return df

    custom_words = list(set(
        list(ENGLISH_STOP_WORDS) + list(stopwords.words('english')) +
        ['california', 'en', 'amp', 'instagram', 'thomas', 'com', 'county', 'org',
         'www', 'https', 'http', 'rt']))

    nlp = spacy.load('en')

    def __init__(self, config_file):
        self.uid = uuid.uuid4()
        self.setup_params = None
        self.disaster_dir = None
        self.twitter_retrieval_dir = None
        self.processed_datasets_dir = None
        self.retrieval_filename = None
        self.processed_filename = None
        self.vectorized_filename = None
        self.clustered_filename = None
        self.modeled_filename = None
        self.retrieval_params = None
        self.processing_params = None
        self.vectorizing_params = None
        self.clustering_params = None
        self.modeling_params = None
        self.config_file = config_file
        self.raw_data = None
        self.disaster_name = None
        self.cluster_model = None
        self.vectorizer = None

        with open(config_file, 'r') as f:
            config = json.load(f)
            print(config)
            print(len(config))
            self.save_params(config)

    def save_params(self, config):
        self.setup_params = config['setup']
        self.dataset = None
        self.disaster_name = self.setup_params['disaster_name']
        self.disaster_dir = self.setup_params['disaster_dir']
        self.twitter_retrieval_dir = self.disaster_dir + '/' + self.setup_params['twitter_retrieval_dir']
        self.processed_datasets_dir = self.disaster_dir + '/' + self.setup_params['processed_datasets_dir']
        self.retrieval_params = config['retrieval']
        self.processing_params = config['processing']
        self.vectorizing_params = config['vectorizing']
        self.clustering_params = config['clustering']
        self.modeling_params = config['modeling']

    def reload_config(self, config_file=None):
        self.uid = uuid.uuid4()
        if config_file is None:
            config_file = self.config_file

        with open(config_file, 'r') as f:
            config = json.load(f)
            self.save_params(config)

    @staticmethod
    def _build_filename(self, *args):
        filename = '_'.join(*args)
        filename += '.csv'
        return filename.replace(" ", "_")

    def retrieve(self):
        params = self.retrieval_params
        search = params["query"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        max_tweets = params["max_tweets"]
        if params['method'] == 'GOT3':
            self.raw_data = self._get_tweets_with_got3(search, start_date, end_date, max_tweets)
            filename = f'{self.disaster_name}_raw_got3' + \
                       f'_{search}_{start_date}_{end_date}_{max_tweets}'
            self.retrieval_filename = f'{self.disaster_name}_raw_got3' + \
                                      f'_{filename}_{self.uid}'.replace(" ", "_")
        else:
            # Twitter Search API
            pass
        self.raw_data.to_csv(f'{self.twitter_retrieval_dir}/{self.retrieval_filename}.csv', index=False)
        return self.raw_data

    def process(self):
        '''Lemmatize using spaCy'''
        if self.raw_data is None:
            self.raw_data = pd.read_csv(self.twitter_retrieval_dir + '/' +
                                        self.retrieval_filename)
        params = self.processing_params
        disaster_specific = params['disaster_specific_stopwords']
        if not params['process_by_user']:
            self.dataset = self._process_data_by_tweet(self.raw_data, disaster_specific)
            file_name = f'{self.disaster_name}_process_by_tweet_{self.processed_filename}.csv'.replace(" ", "_")
        else:
            self.dataset = self._process_data_by_user(self.raw_data, disaster_specific)
        self.processed_data.to_csv(f'{self.processed_datasets_dir}/{file_name}.csv')
        return self.processed_data
        # dataframe = process_data_by_user('../data/mendocinocomplex_pre.csv')

    # features = ['username', 'text']

    def _process_data_by_user(self, file, disaster_specific_stopwords):
        '''aggreggate/lemmatize tweets by user'''
        custom_stopwords = self.custom_words + disaster_specific_stopwords
        df = self.read_file(file)
        # convert to lowercase and remove special chars
        self._remove_extraneous(df, custom_stopwords)
        df['cleaned_tweets'] = self._clean_tweets(df['text'])
        df['lemmatized_tweets'] = self._lemmatize_tweets_spacy(df['cleaned_tweets'])

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

    def _process_data_by_tweet(self, file, disaster_specific_stopwords):
        custom_stopwords = self.custom_words + disaster_specific_stopwords
        df = self.read_file(file)
        # convert to lowercase
        self._remove_extraneous(df, custom_stopwords)
        df['cleaned_tweets'] = self._clean_tweets(df['text'])
        df['lemmatized_tweets'] = self._lemmatize_tweets_spacy(df['cleaned_tweets'])
        return df

    def _lemmatize_tweets_spacy(self, tweet_series):
        lemmas = []
        tokens = []
        for tweet in tweet_series:
            tweet_lemmas, tweet_tokens = self._lemmatize_tweet_spacy(tweet)
            lemmas.append(' '.join(list(tweet_lemmas.values())))
            tokens.append(tokens)
        return pd.Series(lemmas)

    def _lemmatize_tweet_spacy(self, cleaned_tweet):
        # print(f'using spacy on {cleaned_post}')
        doc = self.nlp(cleaned_tweet)
        tokens_lem = [token.lemma_ for token in doc if 'PRON' not in token.lemma_]
        cleaned_post_list = [tl for tl in tokens_lem]
        save_the_lemmas = {t: w for t, w in zip(tokens_lem, cleaned_post_list)}
        return save_the_lemmas, tokens_lem

    def vectorize_tweets(self):
        params = self.vectorizing_params
        print(params)
        if self.dataset is None:
            self.dataset = pd.read_csv(self.processed_datasets_dir + '/' +
                                       self.vectorizing_params['filename'])

        if params['method'] == 'tfidf':
            max_features = params["max_features"]
            ngram_range = params["ngram_range"]
            self.vectorizer = TfidfVectorizer(max_features=eval(max_features), \
                                              ngram_range=eval(ngram_range), \
                                              stop_words=self.custom_words)

            self.vectors = self.vectorizer.fit_transform(self.dataset['lemmatized_tweets'])

            self.vectorized_filename = f'{self.disaster_name}_tfidf' + \
                                       f'_{ngram_range}_{max_features}'.replace(" ", "_")
        elif params['method'] == 'word2vec':
            # word2vec
            tweets = [row.split() for row in self.dataset['lemmatized_tweets']]
            w2v_model = Word2Vec(sentences=tweets,
                                 min_count=1,
                                 window=5,
                                 size=100,
                                 workers=4)
            # window=5,size=100, iterable
            # w2v_model.build_vocab(sentences)
            w2v_model.train(tweets, total_examples=w2v_model.corpus_count,
                            epochs=w2v_model.iter)
            w2v_model.init_sims(replace=True)
            vect_score = []
            for tweet in tweets:
                tweet_vect = []
                for word in tweet:
                    # append the word vector from the model into the tweet_vect list. We will Sum these next.
                    tweet_vect.append(w2v_model.wv[word])
                vect_score.append(sum(tweet_vect) / len(tweet_vect))
            self.vectors = [np.array(f) for f in vect_score]
        return self.vectors
        # elif params['method'] == 'sense2vec':
        #     tweets = [row.split() for row in self.dataset['lemmatized_tweets']]
        #     nlp = spacy.load('en')
        #     s2v_model = Sense2VecComponent()
        #     s2v_model.train(tweets, total_examples=s2v_model.corpus_count, epochs=s2v_model.iter)
        #     s2v_model.init_sims(replace=True)
        #     for tweet in tweets:
        #         tweet_vect = []
        #             # append the word vector from the model into the tweet_vect list. We will Sum these next.
        #         tweet_vect.append(s2v_model.s2v[tweet])
        #     self.dataset['vec_avgs'] = tweet_vect

        #    doc = nlp("A sentence about natural language processing.")

        # self.dataset.to_csv(f'{self.processed_datasets_dir}/{self.vectorized_filename}.csv', index=False)
        # pass

    def cluster(self,vectors):
        if self.clustering_params['method'] == "KMeans_NLTK":
            kmeans = KMeansClusterer(num_means=20, distance=nltk.cluster.util.cosine_distance, repeats=25,
                                     avoid_empty_clusters=True)
            self.dataset['cluster'] = \
                kmeans.cluster(vectors, assign_clusters=True)
        elif self.clustering_params['method'] == "KMeans":
            kmeans = KMeans(n_clusters=eval(self.clustering_params['n_clusters']))
            kmeans.fit(vectors)
            clusters = kmeans.cluster_centers_
            self.dataset['cluster'] = kmeans.predict(self.vectors)
            print(self.dataset[['cluster', 'id', 'text']])
        self.clustered_filename = f'{self.disaster_name}_{self.clustering_params["method"]}' + \
                                  f'_{self.clustering_params["n_clusters"]}'.replace(" ", "_")
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.dataset.to_csv(
            f"{self.disaster_dir}/kmeans/{self.clustering_params['method']}_{self.clustering_params['n_clusters']}_{current_time}.csv",
            index=False)

        filename = f"{self.disaster_dir}/kmeans/{self.clustering_params['method']}_{self.clustering_params['n_clusters']}_{current_time}"

        with open(filename+'.pkl', 'wb') as file:
            pickle.dump(kmeans, file)
            file.close()
        with open(filename+'.vec', 'wb') as file:
            pickle.dump(self.vectors, file)
            file.close()
        return self.dataset, filename+'.pkl'

    def model(self):
        pass

    def output_data(self):
        pass

    def run(self):
        pass

    def load_model(self, model_file):
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        return model

    def count_highest_frequency_words(self, class_vals):
        cv = CountVectorizer()
        sparse_matrix = cv.fit_transform(self.df[class_vals])
        words_features = pd.DataFrame(sparse_matrix.todense(),
                                      columns=cv.get_feature_names())
        word_frequency = words_features.sum()
        word_totals = words_features.sum().sum()
        #     print(word_totals)
        values = word_frequency.sort_values(ascending=False)
        #     print(values/word_totals*100)
        return values
    def temp(self,x):
        try:
            y = eval(x)
        except:
            print(x)
        return y

    def silhouette(self, vectors, model_file=None, filename=None):
        if filename is not None:
            self.dataset = pd.read_csv(filename)
        self.cluster_model = self.load_model(model_file)
        
        if isinstance(self.cluster_model, nltk.cluster.KMeansClusterer):
            n_clusters = self.cluster_model.num_clusters()
        else:
            n_clusters = self.cluster_model.get_params()['n_clusters']
        X = self.dataset.lemmatized_tweets

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        cluster_labels = self.dataset['cluster']

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        # self.dataset['vec_avgs'] = self.dataset['vec_avgs'].apply(lambda x: re.sub("\[\s+", "[", x.strip()))
        # self.dataset['vec_avgs'] = self.dataset['vec_avgs'].apply(lambda x: re.sub("\s+", ",", x.strip()))
        # self.dataset['vec_avgs'] = self.dataset['vec_avgs'].apply(self.temp)

        # test = list(self.dataset['vec_avgs'])

        # count_row = 0
        # for t in test:
        #     count_col = 0
        #     for e in t:
        #         count_col += 1
        #         if not isinstance(e, float):
        #             print(count_row)
        #             print(count_col)
        #     count_row += 1
        # self.dataset['vec_avgs']=self.dataset['vec_avgs'].to_numpy()
        silhouette_avg = silhouette_score(self.vectors, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(self.vectors, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        plt.savefig(
            f'{self.disaster_dir}/kmeans/{self.clustering_params["method"]}_{self.clustering_params["n_clusters"]}')
        plt.show()

        return (plt, fig, ax)


if __name__ == '__main__':
    wf = Workflow('./config/hurricaneharvey_wf_3.json')
    vectors = wf.vectorize_tweets()
    dataset, filename = wf.cluster(vectors)
    model = wf.load_model(filename)

    wf.silhouette(vectors = vectors, model_file="./data/hurricaneharvey/kmeans/KMeans_NLTK_20_2020_06_15_12_16_07.pkl",
                  filename="./data/hurricaneharvey/kmeans/KMeans_NLTK_20_2020_06_15_12_16_07.csv")
