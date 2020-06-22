# Project_5, Problem 11 

Problem Statement: While social media increasingly becomes the first source of information about disasters, this information is not always correct. Posts can be misleading or mistaken. On rarer occasion, users intentionally spread misinformation. How can emergency management agencies identify incorrect information before, during, and after disasters?


## Overview
Natural Disasters are adverse events caused by natural cycles in and on the planet. Some examples would be tornados, earthquakes, volcanic eruptions, hurricanes, tropical storms, wildfires and tsunamis. These events occuring in populated areas can cause major interuptions in ordinary life, and information on things such as evacuations, stay-at-home orders, and best phone numbers to call for help can be life and death. Additionally, false information can muddy the waters to get help and support to the areas that need it most. 

We have been hired by New Light Technologies to create a model that is able to classify a bit of information as either good, bad or neutral means we have to define what those classifciations mean, and then we will need to find examples to show the model what to look for. We will be working with Natural Language Processing for this problem in order to parse through social media messages and determine the meaning behind each message. Then, given what we know to be bad information and good information, we will be able to score each social media post as spreading misinformation or not. 


### The Organization:

>New Light Technologies is a small, award-winning organization based in Washington, D.C. that provides solutions to government, commercial, and non-profit clients. NLT is a team of dedicated technologists, scientists, engineers, designers, and strategists working on some of the most interesting, challenging, and important assignments in the world, ranging from disaster response to enabling growing telecommunications networks to providing healthcare to Americans. Some of the organizations they work with include FEMA (the Federal Emergency Management Agency), USAID (the United States Agency for International Development), the U.S. Census Bureau, and The World Bank.


## Aproach
Initially we thought about what information we have to begin with. We have the information that the agencies are putting out. This is the good information. So we can look at that and set our models to find information like this. The problem is, the information that doesn't align with FEMA and NLT information isn't necessarily bad information. In fact, a majority of it is either good or indifferent. We started by looking at wildfires in California, specifically the Mendocino Complex Fire in 2018. It was one of the worst fires on record. The modeled trained fine, but in all the information that was not like the agency information, there wasn't a single record of misinformation. We applied it to a few other fires, and a few other models, and it was all the same. 

So we decided to change tactics. Instead we needed to find misinformation, and train our model on what that would be. From there, we would have to carry these misinformation ideas from one natrual disaster to the next, as there would be no way to know exactly what sorts of misinformation could spread about a disaster that is currently happening. Here, we looked at Hurricane Harvey, which by doing basic research on google we could find no shortage of misinformation stories. 

## The Data

Initially we explored Facebook and Reddit apis, but we found Twitter to be the best social media platform due to the ease of use of the GotOldTweets api, and the concise and consistent nature of the posts. We built a function that would take a date range and a search term and would return 10,000 tweets at a time. This data also contained information for the number of times each tweets was retweeted, and we were able to run this number as a weight to determine the spread of any particular message. 

### The Botomater
Pulling from our previous cohort at General Assembly, we pulled their Botomater code, and modified it to work with our datasets. This code gave us a ranking on each handle to determine the liklihood of that handle being a bot. Although bots can spread good and bad information, we thought it would useful as a feature in our modeling sequence to help track down misinformation. 

Botomater located here: https://github.com/bwoodhamilton/Social-Media-Misinformation-During-Disasters

### EDA
From here we processed the tweets to remove special characters using RegEx. From there we lemmatized the remaining words in each tweet with SPACY and then tried it through a few vectorizers. After trying CountVectorizer, TFDIF, and Word2Vec. 

### Prodigy Training


## Directory
```bash
project_5  
│ workflow  
│   └──  
│  
├── py_files  
│   └── twitter_retreval.py  
│  
├── jupyter_files  
│   ├── botometer_processing.ipynb  
│   ├── Cluster_EDA_dg.ipynb  
│   ├── Cluster_EDA.ipynb  
│   ├── dg_worksheet_1.ipynb  
│   ├── EDA_dg.ipynb  
│   ├── EDA.ipynb  
│   ├── get_tweets.ipynb  
│   ├── Harvey_Cluster_EDA.ipynb  
│   ├── Initial_Retrieval.ipynb  
│   ├── Kmeans_Result.ipynb  
│   ├── Mendocino_Cluster_EDA.ipynb  
│   ├── Reddit_Thomas_Fire-checkpoint.ipynb  
│   ├── Reddit_Thomas_Fire.ipynb  
│   ├── Untitled.ipynb  
│   ├── w2v_trial_3.ipynb  
│   └── w2v_trial.ipynb  
│
├── images
│   ├── Mendocino_charts
│   │   ├
│
├── data
│   ├── harvey
│   │   ├── hurricaneharvey_retweets.csv
│   │   ├── hurricanharvey_10000.csv
│   │   └── user_scores.csv
│   ├── Mendocino 
│   │   ├── #MendocinoComplex2018-07-262018-10-1.csv
│   │   └── mendocino_retweets.csv
│   ├── Thomasfire
│   │   ├── orgs_thomasfire copy.csv
│   │   ├── orgs_thomasfire.csv
│   │   ├── thomas fire.csv
│   │   ├── Thomas_Hashtag_June_6_2020.csv
│   │   ├── Thomas_Hashtag.csv
│   │   ├── Thomas_initial.csv
│   │   ├── Thomas_Wildfire_California.csv
│   │   └── Thomasfire_June_6_2020.csv
│   └── wildfires
│       ├── bushfire_10000.csv       
│       ├── campfire_10000.csv
│       └── fires_nifc.tsv
└── README.md

```



A clean GitHub repo containing our reproducible results and analysis.
A real-world demonstration of the product.
Any documentation for running the code.
This is an exciting opportunity to identify and solve a problem relevant to the real world problems. Using your data science skills in this practical, pro bono capacity reflects well on you and gives you a great story as you embark on interviews! (The fact that we'll be using open data also gives you free reign to publicly publish your findings and to freely discuss this in interviews.)







├── data
│   └── mendocinocomplex
│       ├── kmeans
│       │   ├── kmeans_20_2020_06_12_15_39_43.pkl
│       │   ├── mendocinocomplex_nclusters_20_2020_06_12_15_39_43.csv
│       │   └── mendocinocomplex_nclusters_7_2020_06_12_02_00_00.csv
│       └── processed_datasets
│           ├── mendocinocomplex.csv
│           ├── mendocinocomplex_pre.csv
│           └── mendocinocomplex_retweets.csv

