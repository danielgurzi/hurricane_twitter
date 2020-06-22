import GetOldTweets3 as got
import datetime

import pandas as pd


def get_tweets(query, start='2006-03-21', end=datetime.date.today().strftime("%Y-%m-%d"), maxtweets=1000):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query) \
        .setSince(start) \
        .setUntil(end) \
        .setMaxTweets(maxtweets)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    tweet_dict = tweetCriteria.__dict__
    file_name = (tweet_dict['querySearch'] + \
                 tweet_dict['since'] + \
                 tweet_dict['until']).replace(" ", "_")
    df = pd.DataFrame([t.__dict__ for t in tweet])
    return df.to_csv(f'./data/{query}_{maxtweets}.csv', index=False)
