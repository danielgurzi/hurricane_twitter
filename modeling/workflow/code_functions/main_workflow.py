import json

from code_functions.process_tweets import process_data_by_tweet, custom_words
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    with open('../code/config/mendocino.json', 'r') as f:
        config = json.load(f)
        params = config
        if params['retrieve_data']:
            # perform data retrieval here
            pass

        if params["process_by_user"]:
            # perform processing by user and model accordingly
            modeling_params = params['modeling']
            file = params['data_dir'] + '/' + modeling_params['filename']

            dataframe = process_data_by_tweet(file)
            tfidf_vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                                        stop_words=custom_words)
            tfidf = tfidf_vec.fit_transform(dataframe['lemmatized_tweets'])
            kmeans = KMeans(n_clusters=20).fit(tfidf)
            # lines_for_predicting = ["tf and idf is awesome!", "some androids is there"]
            # preds = kmeans.predict(tfidf_vec.transform(lines_for_predicting))
            clusters = kmeans.cluster_centers_
            dataframe['predictions'] = kmeans.predict(tfidf)
            print(dataframe[['predictions', 'id', 'text']])
            dataframe.to_csv("../data/labeled_mendocino_20_1000.csv", header=True, index=False)
        else:
            # perform processing by tweet and model accordingly
            pass

