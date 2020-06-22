import json
import GetOldTweets3 as got
import datetime
import pandas as pd
from html import unescape
import uuid
import pandas as pd
import redditcleaner
import regex as re
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from html import unescape
import pickle  # +

import pandas as pd
import numpy as np
import redditcleaner
import regex as re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import Workflow


class GSWorkflow(object):
    """GS workflow"""

    def __init__(self, config_file):
        self.config_file= config_file
        self.uid = uuid.uuid4()
        self.setup_params = None
        self.disaster_dir = None
        self.dataset_dir = None
        self.filename = None
        self.pipeline_params = None
        self.pipeline = None
        self.config_file = config_file
        self.disaster_name = None
        self.gridsearch_json = None
        self.feature_data = None
        self.target_data = None
        self.models_dir = None

        with open(config_file, 'r') as f:
            config = json.load(f)
            self.save_params(self, config)

    def save_params(self, config):
        self.setup_params = config['setup']
        self.disaster_name = self.setup_params['disaster_name']
        self.disaster_dir = self.setup_params['disaster_dir']
        self.dataset_dir = self.disaster_dir + '/' + self.setup_params['dataset_dir']
        self.models_dir = self.disaster_dir + '/' + self.setup_params['models_dir']

        self.gridsearch_json = config['gridsearch']
        self.pipeline = [(name,eval(method)) for name,method in self.gridsearch_json['pipeline']]
        self.pipeline_params = {name:vals for name,vals in self.gridsearch_json["pipeline_params"]}

    def load_data(self):
        pass

    def reload_config(self, config_file=None):
        self.config_file = config_file
        self.uid = uuid.uuid4()
        if config_file is None:
            config_file = self.config_file

        with open(config_file, 'r') as f:
            config = json.load(f)
            self.save_params(config)

    def run_gs_model(self):
        gs_model = GridSearchCV(self.pipeline,  # what object are we optimizing?
                                self.pipeline_params,  # what parameters values are we searching?
                                cv=5)  # 5-fold cross-validation.
        gs_model.fit(self.feature_data, self.target_data)
        best_score = gs_model.best_score_
        best_model = gs_model.best_estimator_

        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_filename = '_'.join([self.config_file, best_score, current_time])
        model_filename = self.models_dir + '/' + model_filename

        with open(model_filename, 'wb') as file:
            pickle.dump(best_model, file)
            file.close()

        return gs_model