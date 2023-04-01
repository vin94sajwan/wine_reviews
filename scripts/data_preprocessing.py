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
data.describe()
data.info()
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

data_part1 = data[["id","country","price","region_1","variety","winery","points","quality"]]
#data_part1.to_csv(r"D:\wine_review\data\wine_quality_part1.csv",index=False)
avg_wines = data[data["quality"].isin(["Under Average wines","Average wines"])]
good_wines = data[data["quality"].isin(["Good wines","Very Good wines","Excellent wines"])]
average = " ".join(avg_wines['cleaned_description'])
good = " ".join(good_wines['cleaned_description'])
corpus =[average,good]
#print(corpus)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,2), max_df = .6, min_df = .01)
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()
dense = X.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
new_data = df.transpose()
new_data.columns = ['average_reviews', 'good_reviews']


# change the value to black
def black_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return ("hsl(0,100%, 1%)")

def wine_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return ("hsl(343, 100%, 39%)")

def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return ("hsl(148, 63%, 31%)")


# instantiate lists of filenames to save pngs, columns to call the wordcloud function on
filenames = ['average_reviews.png', 'good_reviews.png']
columns = list(new_data)
# iterate through inaugural speech vectors to plot and save
mask = np.array(Image.open(r"D:\wine_review\dependencies\_3.jpg"))
os.chdir(r"D:\wine_review\eda")
for i in range(2):
    wordcloud = WordCloud(font_path='C:\Windows\Fonts\Arial.ttf', background_color="white", width=3000,
                          height=2000, max_words=500, mask=mask).generate_from_frequencies(new_data[columns[i]])
    # change the color setting

    if i==0:
        function = wine_color_func
    else:
        function = green_color_func

    wordcloud.recolor(color_func=function)
    plt.figure(figsize=[15, 10])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(filenames[i])
