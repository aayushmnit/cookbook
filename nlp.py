"""
@author: Aayush Agrawal
@Purpose - Re-usable code in Python 3 for Natural Language processing task
"""
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.util import ngrams

def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max+1):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s
    
def make_worlcloud(df,column, bg_color='white', w=600, h=300, font_size_max=100, n_words=40,g_min=1,g_max=1):
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
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()