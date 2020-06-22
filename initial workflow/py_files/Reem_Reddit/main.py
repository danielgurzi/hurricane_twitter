from model import *
from retrieve import *

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords

sub1 = 'writing'
sub2 = 'Screenwriting'
classes = {sub1: 1, sub2: 0}
bow_logreg = 'BOW_LOGREG'
tfidf_logreg = 'TFIDF_LOGREG'
tfidf_multinomial = 'TFIDF_Multinomial'
bow_multinomial = 'BOW_Multinomial'
model_names = [bow_logreg, tfidf_logreg, tfidf_multinomial, bow_multinomial]
models_to_run = {mn: False for mn in model_names}
base_dir = '..'
data_dir = base_dir + '/data'
save_the_lemmas = {}
save_the_stems = {}
custom_words = list(set(
    list(ENGLISH_STOP_WORDS) + list(stopwords.words('english')) + ['really', 'help', 'time', 've', 'write', 'writing',
                                                                   'just', 'like', 'don', "\'t", '-PRON-']))
start_time = datetime(2019, 1, 1, 0, 0)
# end_time = datetime(2019, 3, 15, 0, 0)
end_time = datetime(2019, 2, 15, 0, 0)

start_time_underscore = str(start_time).replace(" ", "_").replace("-", "_").replace(":", "_")
end_time_underscore = str(end_time).replace(" ", "_").replace("-", "_").replace(":", "_")

saved_loc_1 = f'{data_dir}/sub1_{sub1}_{start_time_underscore}_{end_time_underscore}.csv'
saved_loc_2 = f'{data_dir}/sub2_{sub2}_{start_time_underscore}_{end_time_underscore}.csv'


def retrieve_from_reddit(base_dir_inner=base_dir, saved_loc1=saved_loc_1, saved_loc2=saved_loc_2,
                         start=start_time,
                         end=end_time):
    if base_dir_inner == '.' and saved_loc_1[0:2] == '..':
        saved_loc1 = saved_loc_1[1:]
        saved_loc2 = saved_loc_2[1:]
    redd1_dataframe, redd2_dataframe = \
        retrieve_subreddits(sub1, sub2,
                            start,
                            end)

    print(redd1_dataframe.shape)

    redd1_dataframe.drop_duplicates(subset=['created_utc',
                                            'title'], inplace=True)
    redd2_dataframe.drop_duplicates(subset=['created_utc',
                                            'title'], inplace=True)
    print(f'Saving files to:\n sub1: {saved_loc1}\n sub2: {saved_loc2}')

    redd1_dataframe.to_csv(saved_loc1)
    redd2_dataframe.to_csv(saved_loc2)


def run_everything(base_dir_inner, new_run=False,
                   models_to_run_input=models_to_run,
                   lemmatize_bool=True, stemming_bool=True, lemmatize_bool_spacy=True):
    if new_run:
        retrieve_from_reddit(base_dir)

    sub1_dataframe, sub2_dataframe = read_data(base_dir_inner)

    X, y, *_ = prep(sub1_dataframe, sub2_dataframe, sub1, sub2)

    models_scores = model(X, y, models_to_run_input)

    if stemming_bool:
        X, y, _, stems, _ = prep(sub1_dataframe, sub2_dataframe, sub1, sub2, stemming_bool=True)
        models_scores_stemming = model(X, y, models_to_run_input, stemming_bool=True)
        models_scores.update(models_scores_stemming)

    if lemmatize_bool:
        X, y, lemmas, *_ = prep(sub1_dataframe, sub2_dataframe, sub1, sub2, lemmatize_bool=True)
        models_scores_lemmatized = model(X, y, models_to_run_input, lemmatize_bool=True)
        models_scores.update(models_scores_lemmatized)

    if lemmatize_bool_spacy:
        X, y, _, _, lemmas_spacy = prep(sub1_dataframe, sub2_dataframe, sub1, sub2, lemmatize_bool_spacy=True)
        models_scores_lemmatized_spacy = model(X, y, models_to_run_input, lemmatize_bool_spacy=True)
        models_scores.update(models_scores_lemmatized_spacy)

    return models_scores, lemmas, stems, lemmas_spacy


#
if __name__ == '__main__':
    sub1_dataframe, sub2_dataframe = read_data(base_dir)
    X_lemmatized_spacy, y, _, _, lemmas_spacy = prep(sub1_dataframe, sub2_dataframe, sub1, sub2, lemmatize_bool_spacy=True)
    models_scores_lemmatized_spacy = model(X_lemmatized_spacy, y, models_to_run, lemmatize_bool_spacy=True)
    # for mtr in models_to_run.keys():
    #     models_to_run[mtr] = True
    #
    # print(models_to_run)
    #
    # run_scores = run_everything(base_dir_inner='..', new_run=False,
    #                             models_to_run_input=models_to_run,
    #                             lemmatize_bool=False, stemming_bool=False)

    retrieve_from_reddit()

    # run_everything(base_dir=base_dir, new_run=False,
    #                models_to_run_input=models_to_run,
    #                lemmatize_bool=True, stopwords_bool=True)
#     #Should have used argparse here but let it go Reem
#     new_run = str2bool(sys.argv[1])
#     BOW = str2bool(sys.argv[2])
#     TFIDF = str2bool(sys.argv[3])
#     Multinomial = str2bool(sys.argv[4])
#     lemmatize_bool = str2bool(sys.argv[5])
#     stopwords_bool = str2bool(sys.argv[6])
#
#     if new_run:
#         retrieve_from_reddit()
#
#     sub1_dataframe, sub2_dataframe = read_data()
#     X, y = prep(sub1_dataframe, sub2_dataframe, sub1,sub2)
#     models_scores={}
#
#     models_scores = model(X, y, models_to_run)
#
#     if stopwords_bool:
#         X, y = prep(sub1_dataframe, sub2_dataframe, sub1, sub2,stopwords_bool=True)
#         models_scores_stopwords = model(X, y, models_to_run)
#         models_scores.update(models_scores_stopwords)
#
#     if lemmatize_bool:
#         X, y = prep(sub1_dataframe, sub2_dataframe, sub1, sub2,lemmatize_bool)
#         models_scores_lemmatized = model(X, y, models_to_run,lemmatize_bool)
#         models_scores.update(models_scores_lemmatized)
#
#     #return models_scores
