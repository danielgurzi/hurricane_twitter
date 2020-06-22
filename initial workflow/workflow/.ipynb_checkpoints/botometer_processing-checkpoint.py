#!/usr/bin/env python
# coding: utf-8

# # Botometer Processing Notebook
# 
# We decided to use a Python package called to Botometer to help with our analysis. The Botometer is a tool developed by researchers at the Indiana University Network Science Institute (IUNI) and the Center for Complex Networks and Systems Research (CNetS). Scores are displayed as percentages. These percentages are the probability that a twitter account is human or bot; the closer to 0 a score is the higher the likelihood it is a human and the closer to 1 a score is the higher the likelihood it is a bot. According to the Botometer’s website, the “probability calculation uses Bayes’ theorem to take into account an estimate of the overall prevalence of bots, so as to balance false positives with false negatives”.(https://botometer.iuni.iu.edu/#!/faq#what-is-cap) For more information, See Maninder's blog post about the Botometer here: https://medium.com/@m.virk1/botometer-eac76a270516. 

# ### Contents:
# - [Reading in and Inspecting Data](#Reading-in-and-Inspecting-Data)
# - [Getting the Botometer Running](#Getting-the-Botometer-Running)
# - [Making a Usable DataFrame from Botometer Data](#Making-a-Usable-DataFrame-from-Botometer-Data)
# - [Merging, Inspecting, and Preparing the DataFrame](#Merging,-Inspecting,-and-Preparing-the-DataFrame)
# - [Prepping Data for NLP Classification Modeling ](#Prepping-Data-for-NLP-Classification-Modeling)

# In[3]:


get_ipython().system('echo $CONDA_DEFAULT_ENV')


# In[4]:


# Importing packages needed for Data Cleaning and EDA
import os
import pandas as pd 
import matplotlib.pyplot as plt
import botometer


# ### Reading in and Inspecting Data

# In[27]:


# Reading in my proprocessed csv to pandas
twitter = pd.read_csv('./data/hurricaneharvey/twitter_retrieval/hurricaneharvey_10000.csv')


# In[28]:


# Checking the shape of my dataframe 
twitter.shape


# In[29]:


# Seeing what my dataframe looks like 
twitter.head()


# In[30]:


# Seeing how many unique user names there are in my dataframe 
twitter['username'].nunique()


# ### Getting the Botometer Running 

# In[31]:


from dotenv import load_dotenv
load_dotenv()


# In[32]:


# Putting my usernames in a list for processing in the botometer 
username_list = twitter['username'].unique().tolist()


# In[36]:


# Where one would put in their Twitter API credentials and rapid api key and then instantiate a botometer 
rapidapi_key = "4ef98cfe9dmsh457741e02725da2p11bfe4jsnac866de8c63b" # now it's called rapidapi key
twitter_app_auth = {
    'consumer_key' : "Rl7XLgYVy7LYx8A0z848Iu8t3",#os.environ['TWITTER_CONSUMER_KEY'] 
        'consumer_secret' : "SVsuERsHKCC2SzXW1krC7es7RuICItaQc6cJD10AL1DBjrvKyP",#os.environ['TWITTER_CONSUMER_SECRET'] 
        'access_token' : "16955613-1Px1BShX49NFwfk9VpGSJTUHIp8ORrwn3Bc1IOEeY",#os.environ['TWITTER_ACCESS_TOKEN'] 
        'access_secret' : "W69uaFID4GaEEUmNyW0jhB6yOHuWXi8fZTh1gcKbpNiHz",#os.environ['TWITTER_ACCESS_SECRET']  # 
  }
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)


# In[37]:


# Check a sequence of accounts
results = []    
accounts = username_list
for screen_name, result in bom.check_accounts_in(accounts):
    results.append(result)


# In[38]:


# Checking the length of my results to make sure I got what I was expecting 
len(results)


# ### Making a Usable DataFrame from Botometer Data

# In[39]:


# Taking my result list and making it into a dataframe called users_and_scores
# Going through a series of pandas code to make my dataframe into just the username and botrating 
users_and_scores = pd.DataFrame(results)
users_and_scores['cap'] = users_and_scores['cap'].astype(str)
users_and_scores['bot_rating'] = users_and_scores['cap'].str.slice(12,30)
users_and_scores['user'] = users_and_scores['user'].astype(str)
users_and_scores['user'] = [data.split('screen_name')[-1] for data in users_and_scores['user']]
users_and_scores['user'] = users_and_scores['user'].str.replace("'", "")
users_and_scores['user'] = users_and_scores['user'].replace(" ", "")
users_and_scores['user'] = users_and_scores['user'].str.replace(":", "")
users_and_scores['user'] = users_and_scores['user'].str.replace("'", "")
users_and_scores['username'] = users_and_scores['user'].str.replace("}", "")
users_and_scores = users_and_scores.drop(columns=['cap', 'categories', 'display_scores', 'scores', 'user', 'error'])
users_and_scores['bot_rating'] = pd.to_numeric(users_and_scores['bot_rating'], errors='coerce')
users_and_scores.head()


# In[ ]:


# Checking the shape of my dataframe 
users_and_scores.shape


# In[ ]:


# making a file called twitter 2 with the same indexing as I used on my username list
# Reseting the index and eliminating the hashtag in the username
# Saving my work to a csv just in case, also moving the number up by one
twitter2 = twitter
twitter2 = twitter2.reset_index()
twitter2['username'] = twitter2['username'].str.replace('@', '')
twitter2.head()


# ### Merging, Inspecting, and Preparing the DataFrame

# In[ ]:


# Merging my dataframe on the index, also doing .head to make sure the usernames match on both sides 
twitter_bots = twitter2.merge(users_and_scores, left_index=True, right_index=True)
twitter_bots.head()


# In[ ]:


# Doing tail to make sure the usernames match on both sides
twitter_bots.tail()


# In[ ]:


# Checking the shape 
twitter_bots.shape


# In[ ]:


# Dropping unnecessary columns and renaming others, dropping null values, and saving my work to a csv
twitter_bots= twitter_bots.drop(columns=['username_y', 'id', 'link', 'index'])
twitter_bots = twitter_bots.rename(columns={"username_x": "username"})
twitter_bots.dropna(inplace=True)
twitter_bots.head()


# In[ ]:


# Checking the shape after nulls dropped
twitter_bots.shape


# In[ ]:


# Looking at the info 
twitter_bots.info()


# ### Prepping Data for NLP Classification Modeling 

# In[ ]:


# Making one column for text variables, dropping columns, and replacing underscore with a space 
# Saving my work
twitter_bots['words'] = twitter_bots['username'] + ' ' + twitter_bots['hashtags'] + ' ' + twitter_bots['text'] + ' ' + twitter_bots['mentions'] + ' ' + twitter_bots['tweet_to']
twitter_bots.drop(columns=['username', 'text', 'hashtags', 'mentions', 'tweet_to'], inplace=True)
twitter_bots['words'] = twitter_bots['words'].str.replace('_', ' ')
twitter_bots.head()


# In[ ]:


# Dropping duplicates 
twitter_bots = twitter_bots.drop_duplicates()


# In[ ]:


# Checking out the nulls and object types 
twitter_bots.info()


# In[ ]:


# Checking the shape of my dataframe 
twitter_bots.shape


# In[ ]:


# Dropping null values
twitter_bots.dropna(inplace=True)


# In[ ]:


# Making sure all the bot_ratings are numeric, since I made them strings to manipulate the dataframe 
twitter_bots['bot_rating'] = pd.to_numeric(twitter_bots['bot_rating'], errors='coerce')


# In[ ]:


twitter_bots.info()


# In[ ]:


# Seeing how my data looks one last time before saving it to a csv
twitter_bots.head()


# In[ ]:


# Checking the shape one last time 
twitter_bots.shape


# In[ ]:


# Saving my mega dataframe to a csv
twitter_bots.to_csv('./data/twitter_preprocessed_all.csv', index=False)


# In[ ]:




