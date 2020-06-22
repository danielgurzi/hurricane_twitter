import time
import warnings
from datetime import datetime, timezone

import pandas as pd
import requests

warnings.filterwarnings('ignore')


def retrieve_subreddits(subreddit_1, subreddit_2, start=None, end=None):
    sub1_posts = compile_subreddit_dataframe(subreddit_1,
                                             start,

                                             end)

    sub2_posts = compile_subreddit_dataframe(subreddit_2,
                                             start,
                                             end)

    return sub1_posts, sub2_posts


def get_reddit_dataframe(subreddit, after=None, before=None, size=500):
    url = 'https://api.pushshift.io/reddit/search/submission'

    params = {
        'subreddit': subreddit,
        'size': size
    }
    debug_string = f'Retrieving posts from Subreddit: \'{subreddit}\', size: {size}.\n'

    if after:
        params['after'] = int(after.timestamp())
        debug_string += f'After: {after}.\t'
    if before:
        params['before'] = int(before.timestamp())
        debug_string += f'Before: {before}.\t'

    res = requests.get(url, params)
    debug_string += f'\nResponse status code: {res.status_code}'

    while int(res.status_code/100) == 5:
        print(f'Status code: {res.status_code}, trying again after sleep.\n')
        time.sleep(3)
        res = requests.get(url, params)
        debug_string += f'\nResponse status code: {res.status_code}'

    # if res.status_code != 200:
    #     print(f'Status workflow: {res.status_code}\n')
    #     return None

    print(debug_string)
    data = res.json()
    posts = pd.DataFrame(data['data'])
    print(posts.shape)
    return posts


def compile_subreddit_dataframe(subreddit, start=None, end=None):
    large_posts = []
    current = start
    while current < end:
        posts = get_reddit_dataframe(subreddit, current)
        current = datetime.utcfromtimestamp(posts['created_utc'].max())
        if current > end:
            to_remove = posts[pd.to_datetime(posts['created_utc'], utc=True) > end.astimezone(timezone.utc)].index
            posts.drop(to_remove, inplace=True)
        large_posts.append(posts)
        time.sleep(2)

    large_posts = pd.concat(large_posts, axis=0,
                            ignore_index=True).drop_duplicates(subset=['created_utc', 'selftext', 'title'])
    if large_posts.empty:
        print("Error: Empty subreddit posts dataframe {subreddit}")
        return pd.DataFrame()
    return large_posts
