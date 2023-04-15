import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
from PIL import Image
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import os

data = pd.read_csv(r"D:\wine_review\data\winemag-data_first150k.csv", index_col = False)

# set seaborn style
sns.set(style="whitegrid")

stopwords = set(stopwords.words('english'))
# Detokenizer combines tokenized elements
detokenizer = TreebankWordDetokenizer()

def clean_description(desc):
    desc = word_tokenize(desc.lower())
    desc = [token for token in desc if token not in stopwords and token.isalpha()]
    return detokenizer.detokenize(desc)

data["cleaned_description"] = data["description"].apply(clean_description)

print(round((data.isna().sum()/data.shape[0])*100,2))

def wine_quality(column):
    if column < 84:
        quality = "Under Average wines"
    if column >= 84 and column < 88:
        quality = "Average wines"
    if column >= 88 and column < 92:
        quality = "Good wines"
    if column >= 92 and column < 96:
        quality = "Very Good wines"
    if column >= 96:
        quality = "Excellent wines"
    return quality

data.rename(columns={"Unnamed: 0":"id"},inplace=True)
data["quality"]=data["points"].apply(wine_quality)

corpus = " ".join(data['cleaned_description'])
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=.6, min_df=.01)
X = vectorizer.fit_transform(data['cleaned_description'])
terms = vectorizer.get_feature_names()
print(terms)
dense = X.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=terms)
new_df = pd.concat([data[["id","cleaned_description","points","quality"]],df],axis=1)
new_df.to_csv(r"D:\wine_review\data\modelling_data.csv",index=False)

def ranking(corpus):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,2), max_df = .6, min_df = .01)
    X = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names()
    dense = X.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=terms)
    new_data = df.transpose()
    new_data.columns = ['average_reviews', 'good_reviews']

    return new_data
def top_features(data):
    avg_wines = data[data["quality"].isin(["Under Average wines","Average wines"])]
    good_wines = data[data["quality"].isin(["Good wines","Very Good wines","Excellent wines"])]
    average = " ".join(avg_wines['cleaned_description'])
    good = " ".join(good_wines['cleaned_description'])
    corpus =[average,good]
    corpus_ranking = ranking(corpus)
    avg_corpus_ranking = corpus_ranking[["average_reviews"]].reset_index().rename(columns={"index":"features"})
    good_corpus_ranking = corpus_ranking[["good_reviews"]].reset_index().rename(columns={"index":"features"})
    print(avg_corpus_ranking.sort_values("average_reviews",ascending=False))