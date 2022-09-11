import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import contractions
import re
import nltk
from sklearn import preprocessing
import gensim
import string
import multiprocessing
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords, wordnet
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from nltk.stem import PorterStemmer, WordNetLemmatizer

port = PorterStemmer()
lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

printable = set(string.printable)

lemmatized = []

def norm_lemm_v_func(word):
    #words1 = word_tokenize(text)
    text1 = WordNetLemmatizer().lemmatize(word, pos='v')
    lemmatized.append(text1)
    return text1

def word_count_func(text):
    return len(text)

def remove_words(line):
    line = re.sub(" USER", " ", line).strip()
    return re.sub("USER ", " ", line).strip()

import fasttext

pretrained_model = "./fasttext/lid.176.bin" 
model = fasttext.load_model(pretrained_model)

df_olid_train = pd.read_csv("./OLID/olid-training-v1.0.tsv",sep='\t')
df_olid_train.head()

df_olid = pd.read_csv("./OLID/testset-levela.tsv",sep='\t')
df_olid.head()

df_olid_labels = pd.read_csv("./OLID/labels-levela.csv")
df_olid_labels.head()

df_olid_labels.columns = ['id','task']

df_olid_train.dropna()

df_olid_labels.dtypes

df_olid_train.dtypes

olid_test = pd.merge(df_olid, df_olid_labels, on='id')
olid_test.dtypes

olid_test_ab = olid_test

def preprocessing(df):

    df['links_removed'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    
    #df = df.loc[:, ['links_removed']]
    df['special_characters_removed'] = df['links_removed'].map(lambda x: re.sub(r'\W+', ' ', x))
    
    cont = contractions.fix
    #df['no_contract'] = df['special_characters_removed'].apply(lambda x: contractions.fix(x))

    df['no_contract'] = df['special_characters_removed'].apply(lambda x: [contractions.fix(word) for word in x.split()])

    df['no_contract_x'] = df['no_contract'].apply(lambda x: [word for word in x if len(word) > 3])

    df['text_desc'] = [' '.join(map(str, l)) for l in df['no_contract_x']]

    df['removed_user_word'] = df['text_desc'].apply(remove_words)

    langs = []
    for sent in df['removed_user_word']:
        lang = model.predict(sent)[0]
        langs.append(str(lang)[11:13])
    df['langs'] = langs

    df = df[df['langs'] == 'en']
    
    df['tokenized'] = df['removed_user_word'].apply(word_tokenize)
    
    df['lower'] = df['tokenized'].apply(lambda x: [word.lower() for word in x])
    
    punc = string.punctuation
    df['no_punc'] = df['lower'].apply(lambda x: [word for word in x if word not in punc])
    df['cleaned_str'] = [' '.join(map(str, l)) for l in df['no_punc']]
    
    stop_words = set(stopwords.words('english'))
    
    df['stopwords_removed'] = df['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])

    df['lemattized'] = df['stopwords_removed'].apply(lambda x: [norm_lemm_v_func(word) for word in x])
    
    df['stemmetized'] = df['stopwords_removed'].apply(lambda x: [port.stem(word) for word in x])
    
    return df

df = preprocessing(olid_test)
#df.to_csv('./OLID/cleaned_data_olid_test.csv')

frequency_dist = nltk.FreqDist(lemmatized)
frequency_dist.most_common(20)
frequency_dist.plot(30,title="Most common words used",cumulative=False)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(max_font_size=50, max_words=100,
background_color="black").generate_from_frequencies(frequency_dist)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()