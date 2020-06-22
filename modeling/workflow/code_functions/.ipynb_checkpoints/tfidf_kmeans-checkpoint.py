import pickle

import datetime
from code_functions.process_tweets import process_data_by_tweet, custom_words
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    custom_stopwords = [
      "mendocinocomplex",
      "ca",
      "wildfire",
      "mendocinocomplex",
      "wildfires",
      "ventura",
      "montecito",
      "cal",
      "cal_fire"
    ]
    dataframe = process_data_by_tweet('workflow/data/mendocinocomplex/processed_datasets/mendocinocomplex_retweets.csv', custom_stopwords)
    n_clusters = 20
    tfidf_vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                                stop_words=custom_words)
    tfidf = tfidf_vec.fit_transform(dataframe['lemmatized_tweets'])
    kmeans = KMeans(n_clusters=n_clusters).fit(tfidf)
    # lines_for_predicting = ["tf and idf is awesome!", "some androids is there"]
    # preds = kmeans.predict(tfidf_vec.transform(lines_for_predicting))
    clusters = kmeans.cluster_centers_
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"workflow/data/mendocinocomplex/kmeans_{n_clusters}_{current_time}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(kmeans, file)
        file.close()
    dataframe['cluster'] = kmeans.predict(tfidf)
    print(dataframe[['cluster', 'id', 'text']])
    dataframe.to_csv(f"../data/mendocinocomplex/labeled_mendocino_{n_clusters}_{current_time}.csv", header=True, index=False)
