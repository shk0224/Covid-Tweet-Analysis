# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 13:54:43 2021

@author: Shuvo Dutta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords

import os 
os.getcwd()

#1_Introduction and loading dataset
df=pd.read_csv("covid19_tweets.csv")
df
df.head ()
df.shape

needed_columns= ['user_name','date','text']
df=df[needed_columns]
df

df.user_name=df.user_name.astype('category')
df.user_name=df.user_name.cat.codes

df.date=pd.to_datetime(df.date).dt.date
df.head ()

#2_Textprepocessing
texts=df['text']
texts 

removing_url= lambda x: re.sub (r'https\s+', '', str(x))
texts_lr=texts.apply(removing_url)
texts_lr

lower_case=lambda x:x.lower ()
text_lr_lc=texts_lr.apply(lower_case)
text_lr_lc

remove_punctuation= lambda x: x.translate (str.maketrans('','',string.punctuation))
text_lr_lc_np=text_lr_lc.apply(remove_punctuation)
text_lr_lc_np

#3_Lets_explore_data
more_words=['covid','#coronavirus','covid19','cases','coronavirus']
stop_words=set(stopwords.words('English'))
stop_words.update (more_words)

remove_words= lambda x: ' ' .join ([word for word in x.split()if word not in stop_words])
text_lr_lc_np_ns=text_lr_lc_np.apply(remove_words)
text_lr_lc_np_ns


words_list=[word for line in text_lr_lc_np_ns for word in line.split()]
words_list
words_list [:5]

pip install plotly

from collections import Counter

import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

word_counts=Counter(words_list).most_common(20)
words_df=pd.DataFrame(word_counts)
words_df.columns=['word','frequency']
words_df

px.bar(words_df,x='word',y='frequency',title='Most common words')  
                                                              
#or
words_df.plot(x="word", y="frequency", kind="bar").set(title='Most common words')

df.head()
text_lr_lc_np_ns

df.head()
df.text=text_lr_lc_np_ns
df.head()

#4_Stetiment analysis start from here:

!pip install -U textblob
!python -m textblob.download_corpora

pip install vaderSentiment

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()
ps= lambda x:sid.polarity_scores(x)
sentiment_scores=df.text.apply(ps)
sentiment_scores

sentiment_df=pd.DataFrame(data=list(sentiment_scores))
sentiment_df.head()

labelize=lambda x:'neutral' if x==0 else ('positive' if x>0 else 'negative')
sentiment_df['label']=sentiment_df.compound.apply(labelize)
sentiment_df.head()

#5_Visualizing

df.head()
data=df.join(sentiment_df.label)
data.head()

counts_df=data.label.value_counts().reset_index()
counts_df

sns.barplot(x='index',y='label',data=counts_df).set(title='overall view')
#or
px.bar(counts_df,x='index',y='label',title='overall view')

data.head()

data_agg=data [['user_name','date','label']].groupby(['date','label']).count().reset_index()
data_agg.columns=['date','label','counts']
data_agg.head()


px.line(data_agg, x='date',y='counts',color='label',title= 'Daily tweets sentiment')

