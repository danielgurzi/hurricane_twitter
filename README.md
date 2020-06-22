# Project_5, Problem 11

Problem Statement: While social media increasingly becomes the first source of information about disasters, this information is not always correct. Posts can be misleading or mistaken. On rarer occasion, users intentionally spread misinformation. How can emergency management agencies identify incorrect information before, during, and after disasters?


## Overview
Natural Disasters are adverse events caused by natural cycles in and on the planet. Some examples would be tornados, earthquakes, volcanic eruptions, hurricanes, tropical storms, wildfires and tsunamis. These events occuring in populated areas can cause major interuptions in ordinary life, and information on things such as evacuations, stay-at-home orders, and best phone numbers to call for help can be life and death. Additionally, false information can muddy the waters to get help and support to the areas that need it most.

We have been hired by New Light Technologies to create a model that is able to classify a bit of information as either good, bad or neutral means we have to define what those classifciations mean, and then we will need to find examples to show the model what to look for. We will be working with Natural Language Processing for this problem in order to parse through social media messages and determine the meaning behind each message. Then, given what we know to be bad information and good information, we will be able to score each social media post as spreading misinformation or not.


### The Organization:

>New Light Technologies is a small, award-winning organization based in Washington, D.C. that provides solutions to government, commercial, and non-profit clients. NLT is a team of dedicated technologists, scientists, engineers, designers, and strategists working on some of the most interesting, challenging, and important assignments in the world, ranging from disaster response to enabling growing telecommunications networks to providing healthcare to Americans. Some of the organizations they work with include FEMA (the Federal Emergency Management Agency), USAID (the United States Agency for International Development), the U.S. Census Bureau, and The World Bank.


## Aproach
Initially we thought about what information we have to begin with. We have the information that the agencies are putting out. This is the good information. So we can look at that and set our models to find information like this. The problem is, the information that doesn't align with FEMA and NLT information isn't necessarily bad information. In fact, a majority of it is either good or indifferent. We started by looking at wildfires in California, specifically the Mendocino Complex Fire in 2018. It was one of the worst fires on record. The modeled trained fine, but in all the information that was not like the agency information, there wasn't a single record of misinformation. We applied it to a few other fires, and a few other models, and it was all the same.

So we decided to change tactics. Instead we needed to find misinformation, and train our model on what that would be. From there, we would have to carry these misinformation ideas from one natural disaster to the next, as there would be no way to know exactly what sorts of misinformation could spread about a disaster that is currently happening. Here, we looked at Hurricane Harvey, which by doing basic research on google we could find no shortage of misinformation stories.

## The Data

Initially we explored Facebook and Reddit apis, but we found Twitter to be the best social media platform due to the ease of use of the GotOldTweets api, and the concise and consistent nature of the posts. We built a function that would take a date range and a search term and would return 10,000 tweets at a time. This data also contained information for the number of times each tweets was retweeted, and we were able to run this number as a weight to determine the spread of any particular message.

### The Botometer
Pulling from our previous cohort at General Assembly, we pulled their Botomater code, and modified it to work with our datasets. This code gave us a ranking on each handle to determine the liklihood of that handle being a bot. Although bots can spread good and bad information, we thought it would useful as a feature in our modeling sequence to help track down misinformation.

Botomater located here: https://github.com/bwoodhamilton/Social-Media-Misinformation-During-Disasters

### EDA
From here we processed the tweets to remove special characters using RegEx. From there we lemmatized the remaining words in each tweet with SPACY and then tried it through a few vectorizers. After trying CountVectorizer, TFDIF, and Word2Vec (see: data/data_generation_eda/Harvey_Cluster_EDA).

### KMeans

We attempted multiple runs of KMeans as a means of performing EDA as well and some preliminary clustering (see Workflow_Mod.py and Kmeans_Result).

### Prodigy Training
Training took place between two different members. NER was performed using seeds. All entities relevant to the custom label HOAX were annotated using prodigy, then a model was generated (see NER directory). Then we performed text classification (see TextClassification/TextClassification_NER.ipynb).

## Directory
```bash
.
├── README.md
├── data
│   ├── data_generation_eda
│   │   ├── Harvey_Cluster_EDA.ipynb
│   │   ├── Kmeans_Result.ipynb
│   │   ├── trials/
│   │   └── w2v_trial.ipynb
│   ├── harvey
│   │   ├── df28.csv
│   │   ├── harvey_hoax.csv
│   │   ├── hurricaneharvey_10000.csv
│   │   ├── hurricaneharvey_18000.csv
│   │   └── hurricaneharvey_retweets.csv
│   ├── hurricaneharvey
│   │   ├── processed_datasets
│   │   │   ├── hurricaneharvey_10000.csv
│   │   │   └── hurricaneharvey_tfidf_(1,_2)_1000.csv
│   │   ├── twitter_retrieval
│   │   │   ├── hurricaneharvey_10000.csv
│   │   │   └── hurricaneharvey_retweets_10000.csv
│   │   ├── user_scores_botometer.csv
│   │   └── vectorized_datasets
│   ├── hurricanemaria
│   │   └── hurricanemaria_10000.csv
│   ├── irma
│   │   └── hurricaneirma_10000.csv
└── modeling
    ├── NER
    │   ├── RedditHandler.py
    │   ├── SpacyProcessor.ipynb
    │   ├── conspiracy01_1000.jsonl
    │   ├── conspiracy02_1000.jsonl
    │   ├── conspiracy_3000.jsonl
    │   ├── hoax_patterns.jsonl
    │   ├── hurricaneharvey_10000_tweets.jsonl
    │   ├── hurricaneharvey_1000_cleaned_tweets.jsonl
    │   ├── seeds.txt
    │   ├── shuffled_harvey.jsonl
    │   ├── spacy_formatter.py
    │   └── the_donald_1000.jsonl
    ├── TextClassification
    │   ├── TextClassification_NER.ipynb
    │   └── commands_prodigy_project5.txt
    ├── kmeans
    │   ├── KMeans_20.png
    │   ├── KMeans_20_2020_06_15_18_30_40.csv
    │   ├── KMeans_20_2020_06_15_18_30_40.pkl
    │   ├── KMeans_20_2020_06_15_18_30_40.vec
    │   ├── KMeans_20_2020_06_15_18_31_53.csv
    │   ├── KMeans_20_2020_06_15_18_31_53.pkl
    │   ├── KMeans_20_2020_06_15_18_31_53.vec
    │   ├── KMeans_20_2020_06_15_18_32_38.csv
    │   ├── KMeans_20_2020_06_15_18_32_38.pkl
    │   ├── KMeans_20_2020_06_15_18_32_38.vec
    │   ├── KMeans_20_2020_06_15_19_02_32.csv
    │   ├── KMeans_20_2020_06_15_19_02_32.pkl
    │   ├── KMeans_20_2020_06_15_19_02_32.vec
    │   ├── KMeans_20_2020_06_15_19_03_09.csv
    │   ├── KMeans_20_2020_06_15_19_03_09.pkl
    │   ├── KMeans_20_2020_06_15_19_03_09.vec
    │   ├── KMeans_20_2020_06_15_19_05_26.csv
    │   ├── KMeans_20_2020_06_15_19_05_26.pkl
    │   ├── KMeans_20_2020_06_15_19_05_26.vec
    │   ├── KMeans_20_2020_06_15_19_06_22.csv
    │   ├── KMeans_20_2020_06_15_19_06_22.pkl
    │   ├── KMeans_20_2020_06_15_19_06_22.vec
    │   ├── KMeans_20_2020_06_15_19_07_36.csv
    │   ├── KMeans_20_2020_06_15_19_07_36.pkl
    │   ├── KMeans_20_2020_06_15_19_07_36.vec
    │   ├── KMeans_20_2020_06_15_19_11_17.csv
    │   ├── KMeans_20_2020_06_15_19_11_17.pkl
    │   ├── KMeans_20_2020_06_15_19_11_17.vec
    │   ├── KMeans_20_2020_06_15_19_13_26.csv
    │   ├── KMeans_20_2020_06_15_19_13_26.pkl
    │   ├── KMeans_20_2020_06_15_19_13_26.vec
    │   ├── KMeans_20_2020_06_15_19_14_04.csv
    │   ├── KMeans_20_2020_06_15_19_14_04.pkl
    │   ├── KMeans_20_2020_06_15_19_14_04.vec
    │   ├── KMeans_20_2020_06_15_19_14_32.csv
    │   ├── KMeans_20_2020_06_15_19_14_32.pkl
    │   ├── KMeans_20_2020_06_15_19_14_32.vec
    │   ├── KMeans_20_2020_06_15_19_39_42.csv
    │   ├── KMeans_20_2020_06_15_19_39_42.pkl
    │   ├── KMeans_20_2020_06_15_19_39_42.vec
    │   ├── KMeans_20_2020_06_15_19_41_39.csv
    │   ├── KMeans_20_2020_06_15_19_41_39.pkl
    │   ├── KMeans_20_2020_06_15_19_41_39.vec
    │   ├── KMeans_20_2020_06_15_19_43_09.csv
    │   ├── KMeans_20_2020_06_15_19_43_09.pkl
    │   ├── KMeans_20_2020_06_15_19_43_09.vec
    │   ├── KMeans_20_2020_06_15_19_44_18.csv
    │   ├── KMeans_20_2020_06_15_19_44_18.pkl
    │   ├── KMeans_20_2020_06_15_19_44_18.vec
    │   ├── KMeans_20_2020_06_15_19_44_58.csv
    │   ├── KMeans_20_2020_06_15_19_44_58.pkl
    │   ├── KMeans_20_2020_06_15_19_44_58.vec
    │   ├── KMeans_20_2020_06_15_19_45_26.csv
    │   ├── KMeans_20_2020_06_15_19_45_26.pkl
    │   ├── KMeans_20_2020_06_15_19_45_26.vec
    │   ├── KMeans_20_2020_06_15_19_47_34.csv
    │   ├── KMeans_20_2020_06_15_19_47_34.pkl
    │   ├── KMeans_20_2020_06_15_19_47_34.vec
    │   ├── KMeans_NLTK_20_2020_06_14_06_34_36.pkl
    │   ├── KMeans_NLTK_20_2020_06_14_17_27_19.pkl
    │   ├── KMeans_NLTK_20_2020_06_14_18_01_01.pkl
    │   ├── KMeans_NLTK_20_2020_06_14_18_04_50.pkl
    │   ├── KMeans_NLTK_20_2020_06_14_18_37_13.pkl
    │   ├── KMeans_NLTK_20_2020_06_15_10_33_46.csv
    │   ├── KMeans_NLTK_20_2020_06_15_10_33_46.pkl
    │   ├── KMeans_NLTK_20_2020_06_15_12_16_07.csv
    │   ├── KMeans_NLTK_20_2020_06_15_12_16_07.pkl
    │   ├── hurricaneharvey_20_2020_06_13_04_38_43.csv
    │   ├── hurricaneharvey_20_2020_06_13_04_44_36.csv
    │   ├── kmeans_20_2020_06_13_04_38_43.pkl
    │   ├── kmeans_20_2020_06_13_04_44_36.pkl
    │   ├── kmeans_20_2020_06_15_19_20_04.png
    │   └── kmeans_20_2020_06_15_19_48_17.png
    └── workflow
        ├── __init__.py
        ├── code_functions
        │   ├── get_tweets.py
        │   ├── main_workflow.py
        │   ├── process_tweets.py
        │   ├── tfidf_kmeans.py
        │   └── word2vec_kmeans.py
        └── trials
            ├── Bunch.py
            ├── GSWorkflow.py
            ├── Workflow_Mod.py
            ├── botometer_processing.py
            └── config
                ├── hurricaneharvey_gs_1.json
                ├── hurricaneharvey_wf_2.json
                ├── hurricaneharvey_wf_3.json
                ├── hurricaneharvey_wf_4.json
                └── mendocino.json

```
