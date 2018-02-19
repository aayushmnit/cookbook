"""
@author: Aayush Agrawal
@Purpose - Re-usable code in Python 3 for Natural Language processing task
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

eng_stop = set(stopwords.words('english'))


def word_grams(text, min=1, max=4):
    '''
    Function to create N-grams from text
    Required Input -
        - text = text string for which N-gram needs to be created
        - min = minimum number of N
        - max = maximum number of N
    Expected Output -
        - s = list of N-grams 
    '''
    s = []
    for n in range(min, max+1):
        for ngram in ngrams(text, n):
            s.append(' '.join(str(i) for i in ngram))
    return s
    
def make_worlcloud(df,column, bg_color='white', w=1200, h=1000, font_size_max=50, n_words=40,g_min=1,g_max=1):
    '''
    Function to make wordcloud from a text corpus
    Required Input -
        - df = Pandas DataFrame
        - column = name of column containing text
        - bg_color = Background color
        - w = width
        - h = height
        - font_size_max = maximum font size allowed
        - n_word = maximum words allowed
        - g_min = minimum n-grams
        - g_max = maximum n-grams
    Expected Output -
        - World cloud image
    '''
    text = ""
    for ind, row in df.iterrows(): 
        text += row[column] + " "
    text = text.strip().split(' ') 
    text = word_grams(text,g_min,g_max)
    
    text = list(pd.Series(word_grams(text,1,2)).apply(lambda x: x.replace(' ','_')))
    
    s = ""
    for i in range(len(text)):
        s += text[i] + " "

    wordcloud = WordCloud(background_color=bg_color, \
                          width=w, \
                          height=h, \
                          max_font_size=font_size_max, \
                          max_words=n_words).generate(s)
    wordcloud.recolor(random_state=1)
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    

def get_tokens(text):
    '''
    Function to tokenize the text
    Required Input - 
        - text - text string which needs to be tokenized
    Expected Output -
        - text - tokenized list output
    '''
    return word_tokenize(text)

def convert_lowercase(text):
    '''
    Function to tokenize the text
    Required Input - 
        - text - text string which needs to be lowercased
    Expected Output -
        - text - lower cased text string output
    '''
    return text.lower()

def remove_punctuations(text):
    '''
    Function to tokenize the text
    Required Input - 
        - text - text string 
    Expected Output -
        - text - text string with punctuation removed
    '''
    return text.translate(None,string.punctuation)

def remove_stopwords(text):
    '''
    Function to tokenize the text
    Required Input - 
        - text - text string which needs to be tokenized
    Expected Output -
        - text - list output with stopwords removed
    '''
    return [word for word in text.split() if word not in eng_stop]
    
def convert_stemmer(word):
    '''
    Function to tokenize the text
    Required Input - 
        - word - word which needs to be tokenized
    Expected Output -
        - text - word output after stemming
    '''
    porter_stemmer = PorterStemmer()
    return porter_stemmer.stem(word)

def convert_lemmatizer(word):
    '''
    Function to tokenize the text
    Required Input - 
        - word - word which needs to be lemmatized
    Expected Output -
        - word - word output after lemmatizing
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    return wordnet_lemmatizer.lemmatize(word)
    
def create_tf_idf(df, column, train_df = None, test_df = None,n_features = None):
    '''
    Function to do tf-idf on a pandas dataframe
    Required Input -
        - df = Pandas DataFrame
        - column = name of column containing text
        - train_df(optional) = Train DataFrame
        - test_df(optional) = Test DataFrame
        - n_features(optional) = Maximum number of features needed
    Expected Output -
        - train_tfidf = train tf-idf sparse matrix output
        - test_tfidf = test tf-idf sparse matrix output
        - tfidf_obj = tf-idf model
    '''
    tfidf_obj = TfidfVectorizer(ngram_range=(1,1), stop_words='english', 
                                analyzer='word', max_features = n_features)
    tfidf_text = tfidf_obj.fit_transform(df.ix[:,column].values)
    
    if train_df is not None:        
        train_tfidf = tfidf_obj.transform(train_df.ix[:,column].values)
    else:
        train_tfidf = tfidf_text

    test_tfidf = None
    if test_df is not None:
        test_tfidf = tfidf_obj.transform(test_df.ix[:,column].values)

    return train_tfidf, test_tfidf, tfidf_obj
    
def create_countvector(df, column, train_df = None, test_df = None,n_features = None):
    '''
    Function to do count vectorizer on a pandas dataframe
    Required Input -
        - df = Pandas DataFrame
        - column = name of column containing text
        - train_df(optional) = Train DataFrame
        - test_df(optional) = Test DataFrame
        - n_features(optional) = Maximum number of features needed
    Expected Output -
        - train_cvect = train count vectorized sparse matrix output
        - test_cvect = test count vectorized sparse matrix output
        - cvect_obj = count vectorized model
    '''
    cvect_obj = CountVectorizer(ngram_range=(1,1), stop_words='english', 
                                analyzer='word', max_features = n_features)
    cvect_text = cvect_obj.fit_transform(df.ix[:,column].values)
    
    if train_df is not None:
        train_cvect = cvect_obj.transform(train_df.ix[:,column].values)
    else:
        train_cvect = cvect_text
        
    test_cvect = None
    if test_df is not None:
        test_cvect = cvect_obj.transform(test_df.ix[:,column].values)

    return train_cvect, test_cvect, cvect_obj