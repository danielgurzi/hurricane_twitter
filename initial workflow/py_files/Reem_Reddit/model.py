from html import unescape

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

from main import sub1, sub2, bow_logreg, tfidf_logreg, tfidf_multinomial, bow_multinomial, saved_loc_1, saved_loc_2, \
    base_dir, custom_words

nlp = spacy.load('en', disable=['parser', 'ner'])
pipes_dict = dict()
pipe_params = dict()

pipes_dict[bow_logreg] = Pipeline([
    ('cvec', CountVectorizer()),
    ('logreg', LogisticRegression())
])
pipe_params[bow_logreg] = {
    'cvec__max_features': [1000, 5000, 10000],
    'cvec__ngram_range': [(1, 1), (1, 2)],
    'cvec__stop_words': [None, 'english', custom_words],
    'logreg__max_iter': [500] * 3
}

pipes_dict[tfidf_logreg] = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression())
])

pipe_params[tfidf_logreg] = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__stop_words': [None, 'english', custom_words],
    'logreg__max_iter': [500] * 3
}

pipes_dict[tfidf_multinomial] = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('multi', MultinomialNB())
])

pipe_params[tfidf_multinomial] = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__stop_words': [None, 'english', custom_words]
}

pipes_dict[bow_multinomial] = Pipeline([
    ('cvec', CountVectorizer()),
    ('multi', MultinomialNB())
])
pipe_params[bow_multinomial] = {
    'cvec__max_features': [1000, 5000, 10000],
    'cvec__ngram_range': [(1, 1), (1, 2)],
    'cvec__stop_words': [None, 'english', custom_words],
}


def read_data(base_dir_inner=base_dir, saved_loc1=saved_loc_1, saved_loc2=saved_loc_2):
    if base_dir_inner == '.' and saved_loc1[0:2] == '..':
        saved_loc1 = saved_loc1[1:]
        saved_loc2 = saved_loc2[1:]

    print(f'Reading data files:\nsub_1:{saved_loc1}\nsub_2:{saved_loc2}')
    sub1_reddit = pd.read_csv(saved_loc1)
    sub2_reddit = pd.read_csv(saved_loc2)
    print(f'sub1_reddit: {sub1}.\nRows:{sub1_reddit.shape[0]}')
    print(f'sub2_reddit: {sub2}.\nRows:{sub2_reddit.shape[0]}')
    return sub1_reddit, sub2_reddit


def combine_title_with_text(dataframe):
    dataframe['posts'] = dataframe[['title', 'selftext']].agg('-'.join, axis=1)
    return dataframe


def posts_to_words(raw_post):
    raw_post = redditcleaner.clean(raw_post)
    post_text = BeautifulSoup(unescape(raw_post)).get_text()
    alpha_characters_only = re.sub("[^a-zA-Z]+", " ", post_text)
    alpha_characters_only = alpha_characters_only.replace("&", ' ').replace("#x200B", ' ').replace('[', ' ').replace(
        ']', ' ')

    words = alpha_characters_only.lower().split()
    return words


def process_text(dataframe, name,
                 lemmatize_bool=False, stemming_bool=False, lemmatize_spacy_bool=False):
    the_lemmas = {}
    the_stems = {}
    the_spacy_lemmas = {}
    clean_posts = []
    print(f"Cleaning and parsing the {name} set reddit posts...")
    total_posts = dataframe.shape[0]
    # Instantiate counter.
    j = 0
    print(f'total posts: {total_posts}')
    # For every review in our training set...
    for post in dataframe['posts']:
        # print(f'post contents: {post}')
        # Convert review to words, then append to clean_train_reviews.
        cleaned_post_list = posts_to_words(post)
        # if stopwords_bool:
        #     stops = set(custom_words)
        #     # cleaned_post=cleaned_post.lower().split()
        #     cleaned_post_list = [w for w in cleaned_post_list if w not in stops]
        # cleaned_post = " ".join(meaningful_words)
        if lemmatize_spacy_bool:
            post_lemmas, cleaned_post_list = lemmatize_posts_spacy(' '.join(cleaned_post_list))
            the_spacy_lemmas.update(post_lemmas)
        elif lemmatize_bool:
            # post_lemmas, cleaned_post_list = lemmatize_posts_spacy(' '.join(cleaned_post_list))
            post_lemmas, cleaned_post_list = lemmatize_posts(cleaned_post_list)
            the_lemmas.update(post_lemmas)
        if stemming_bool:
            post_stems, cleaned_post_list = stem_posts(cleaned_post_list)
            the_stems.update(post_stems)
        cleaned_post = " ".join(cleaned_post_list)
        clean_posts.append(cleaned_post)

        # If the index is divisible by 1000, print a message.
        if (j + 1) % 1000 == 0:
            print(f'Post {j + 1} of {total_posts}.')
        j += 1
    return clean_posts, the_lemmas, the_stems, the_spacy_lemmas


def preprocess(dataframe, subreddit_name,
               lemmatize_bool=False, stemming_bool=False, lemmatize_bool_spacy=False):
    dataframe[['title', 'selftext']] = \
        dataframe[['title', 'selftext']].fillna('')
    dataframe['posts'] = \
        dataframe[['title', 'selftext']].agg(' '.join, axis=1)
    dataframe.replace("", np.nan, inplace=True)
    dataframe.dropna(subset=['posts'], inplace=True)
    words, lemmas, stems, spacy_lemmas = process_text(dataframe, subreddit_name,
                                                      lemmatize_bool, stemming_bool, lemmatize_bool_spacy)
    return dataframe, words, lemmas, stems, spacy_lemmas


def prep(dataframe1, dataframe2, sub1_name, sub2_name,
         lemmatize_bool=False,
         stemming_bool=False,
         lemmatize_bool_spacy=False):

    print('+' * 80)
    print(f"Lemmatize:{lemmatize_bool}, Stemming:{stemming_bool}, spaCy Lemmatize:{lemmatize_bool_spacy}")
    dataframe1, words_1, lemmas_1, stems_1, spacy_lemmas_1 = preprocess(dataframe1, sub1_name,
                                                                        lemmatize_bool, stemming_bool,
                                                                        lemmatize_bool_spacy)
    dataframe2, words_2, lemmas_2, stems_2, spacy_lemmas_2 = preprocess(dataframe2, sub2_name,
                                                                        lemmatize_bool, stemming_bool,
                                                                        lemmatize_bool_spacy)
    lemmas = lemmas_1
    lemmas.update(lemmas_2)
    stems = stems_1
    stems.update(stems_2)
    spacy_lemmas = spacy_lemmas_1
    spacy_lemmas.update(spacy_lemmas_2)

    y_1 = [1] * len(words_1)
    y_2 = [0] * len(words_2)
    y = pd.Series(y_1 + y_2)
    words = words_1 + words_2
    X = pd.Series(words)
    return X, y, lemmas, stems, spacy_lemmas


def run_model(model_name, X_train, y_train, X_test, y_test):
    gs_model = GridSearchCV(pipes_dict[model_name],  # what object are we optimizing?
                            pipe_params[model_name],  # what parameters values are we searching?
                            cv=5)  # 5-fold cross-validation.
    print(f"Fitting training set using {model_name}")
    gs_model.fit(X_train, y_train)
    train_score, test_score = print_scores(gs_model, X_train, y_train, X_test, y_test)
    return {'model': gs_model, 'best_params': gs_model.best_params_, 'train_score': train_score,
            'test_score': test_score}


def model(X, y, models, lemmatize_bool=False, stemming_bool=False, lemmatize_bool_spacy=False):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        stratify=y,
                                                        random_state=42)
    models_scores = {}
    for model_name, model_bool in models.items():
        model_run = run_model(model_name, X_train, y_train, X_test, y_test)
        if model_bool:
            if lemmatize_bool:
                models_scores[model_name + '_lemmatized'] = model_run
            elif stemming_bool:
                models_scores[model_name + '_stemming'] = model_run
            elif lemmatize_bool_spacy:
                models_scores[model_name + '_lemmatized_spacy'] = model_run
            else:
                models_scores[model_name] = model_run

    return models_scores


def stem_posts(cleaned_post_list):
    ps = SnowballStemmer("english")
    stemmed_words = [ps.stem(w) for w in cleaned_post_list]
    save_the_stems = {t: w for t, w in zip(stemmed_words, cleaned_post_list)}
    return save_the_stems, stemmed_words


def lemmatize_posts_spacy(cleaned_post):
    # print(f'using spacy on {cleaned_post}')
    doc = nlp(cleaned_post)
    tokens_lem = [token.lemma_ for token in doc if 'PRON' not in token.lemma_]
    cleaned_post_list = [tl for tl in tokens_lem]
    save_the_lemmas = {t: w for t, w in zip(tokens_lem, cleaned_post_list)}
    return save_the_lemmas, tokens_lem


def lemmatize_posts(cleaned_post_list):
    # Import lemmatizer. (Same as above.)
    lemmatizer = WordNetLemmatizer()
    tokens_lem = [lemmatizer.lemmatize(i) for i in cleaned_post_list]
    save_the_lemmas = {t: w for t, w in zip(tokens_lem, cleaned_post_list)}
    return save_the_lemmas, tokens_lem


def print_scores(estimator, X_train, y_train, X_test, y_test):
    score_train = estimator.best_estimator_.score(X_train, y_train)
    score_test = estimator.best_estimator_.score(X_test, y_test)
    print(f"Train score: {score_train}, Test score: {score_test}")
    # print(f"Best Params: {estimator.best_params_}")
    return score_train, score_test
