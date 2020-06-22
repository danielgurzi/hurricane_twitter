import pickle

import datetime
from code_functions.process_tweets import process_data_by_tweet, custom_words
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from process_tweets import custom_words

if __name__ == '__main__':
    disaster_specific = ['hurricaneharvey', 'harvey', 'houston', 'texas']
    custom_stopwords = custom_words + disaster_specific

    dataframe = process_data_by_tweet('../data/mendocinocomplex/twitter_retrieval/mendocinocomplex.csv', custom_stopwords)
    dataframe.to_csv('../data/mendocinocomplex/processed_datasets/mendocinocomplex_retweets.csv')
    n_clusters = 20
    tfidf_vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                                stop_words=custom_words)
    tfidf = tfidf_vec.fit_transform(dataframe['lemmatized_tweets'])
    kmeans = KMeans(n_clusters=n_clusters).fit(tfidf)
    # lines_for_predicting = ["tf and idf is awesome!", "some androids is there"]
    # preds = kmeans.predict(tfidf_vec.transform(lines_for_predicting))
    clusters = kmeans.cluster_centers_
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"../data/mendocinocomplex/kmeans/kmeans_{n_clusters}_{current_time}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(kmeans, file)
        file.close()
    dataframe['cluster'] = kmeans.predict(tfidf)
    print(dataframe[['cluster', 'id', 'text']])
    dataframe.to_csv(f"../data/mendocinocomplex/kmeans/mendocinocomplex_{n_clusters}_{current_time}.csv", header=True, index=False)
