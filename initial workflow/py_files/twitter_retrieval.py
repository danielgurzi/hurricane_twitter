'''

Code from https://towardsdatascience.com/mining-twitter-data-ba4e44e6aecc
'''

import os

import sys
from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler


def get_twitter_auth():
    # set up twitter authentication
    # Return: tweepy.OAuthHandler object
    try:
        consumer_key = "Rl7XLgYVy7LYx8A0z848Iu8t3"#os.environ['TWITTER_CONSUMER_KEY'] 
        consumer_secret = "SVsuERsHKCC2SzXW1krC7es7RuICItaQc6cJD10AL1DBjrvKyP"#os.environ['TWITTER_CONSUMER_SECRET'] 
        access_token = "16955613-1Px1BShX49NFwfk9VpGSJTUHIp8ORrwn3Bc1IOEeY"#os.environ['TWITTER_ACCESS_TOKEN'] 
        access_secret = "W69uaFID4GaEEUmNyW0jhB6yOHuWXi8fZTh1gcKbpNiHz"#os.environ['TWITTER_ACCESS_SECRET']  # 
    except KeyError:
        sys.stderr.write("TWITTER_* environment variables not set\n")
        sys.exit(1)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth


def get_twitter_client():
    # Setup twitter API client.
    # Return tweepy.API object
    auth = get_twitter_auth()
    client = API(auth)
    return client


def get_tweets(tweepy_api, search_words, date_since=None, date_until=None):
    # Collect tweets
    # tweets = Cursor(tweepy_api.search,
    #                 q=search_words,
    #                 lang="en",
    #                 since=date_since, until=date_until).items(5)

    tweets = Cursor(tweepy_api.search,
                    q=search_words,
                    lang="en").items(5)
    # Iterate and print tweets
    print('Printing Tweets:')

    for tweet in tweets:
        print(f'Created: {tweet.created_at}.\n Text:{tweet.text}')
        tweet_dict = vars(tweet)
        print(tweet_dict.keys())
        print(tweet_dict['_json']['geo'])
        print(tweet_dict['geo'])
        print(tweet_dict['_json'])
        print(tweet_dict['coordinates'])

        # print(vars(tweet)['geo'])
        # if tweet.geo_enabled:
        #     print(tweet.coordinates)


if __name__ == "__main__":
    api = get_twitter_client()
    # test authentication
    try:
        api.verify_credentials()
        print("Authentication OK")
    except Exception as e:
        print("Error during authentication")
        print(e)

    get_tweets(api, "Mendocino")
