# Let's import our libraries
import pandas as pd
import numpy as np
import seaborn as sns
import praw
# import regex as re

class reddit_getter:
    def __init__(self, sub):
        # This initalizes and loads all our past datasets
        self.subreddit = sub
        self.submits = 1000
        self.time_filter = 'day'
        self.reddit = praw.Reddit(client_id='oZRFffq6V4xYMg',
                                  client_secret='O_p5-j5D3fYEa58nqMOwHnGZw0E',
                                  username='the_illuminati_666',
                                  user_agent='nothing_nefarious')
        self.dframe = pd.read_csv('../data_files/combined_reddit_data.csv')

    # This function can be called to pull data from Reddit, provided a submission_getter object has been created
    def get(self):
        comment_dict = {}
        subreddit = self.reddit.subreddit(self.subreddit)
        reddit_data = subreddit.top(time_filter=self.time_filter, limit=self.submits)
        print(f'Subreddit We Are Mining: {self.subreddit}, Time Filter: {self.time_filter}')
        for submission in reddit_data:
            # sometimes the submission.stickied method throws an error, we'll use a try and except in case
            try:
                if not submission.stickied:
                    comment_dict.update({submission.id: [submission.title, submission.author, subreddit.title,
                                                         submission.num_comments, str(self.time_filter)]})
            except:
                pass
        df = pd.DataFrame.from_dict(data=comment_dict, orient='index',
                                    columns=['Submission Title', 'Submission Author', 'Subreddit', 'Number of Comments',
                                             'Time Filter'])
        print('Returning DataFrame')
        return df