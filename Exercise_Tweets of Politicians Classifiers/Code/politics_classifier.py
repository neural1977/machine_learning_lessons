'''
Created on 26/09/2019

@author: Ufuk Kaliscan, Francesco Pugliese
'''

import pdb
import tweepy
import csv
import pandas as pd
import numpy as np
import re
import sys
import json
import nltk
import string
import pattern.it
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from urllib.request import urlopen

def lemmatize_word(input_word):
    in_word = input_word
    word_it = pattern.it.parse(
        in_word, 
        tokenize=False,  
        tag=False,  
        chunk=False,  
        lemmata=True 
    )
    the_lemmatized_word = word_it.split()[0][0][4]
    return the_lemmatized_word

def cleaner(s):
    
    a=re.sub('#\S*','',s)
    a=re.sub('https\S*','',a)
    a = re.sub(r'[^\w\s]','',a)
    
    a=a.strip()
    if (a=='') or ('RT ' in s):
        return np.NaN
    else:
        # 1st tokenize the sentence(s)
        word_tokenized_list = nltk.tokenize.word_tokenize(a)
        # 2nd remove punctuation and everything lower case
        word_tokenized_no_punct = [x.lower() for x in word_tokenized_list if x not in string.punctuation]
        # 3rd remove stop words (for the Italian language)
        word_tokenized_no_punct_no_sw = [x for x in word_tokenized_no_punct if x not in it_stop_words]
        # 4th lemmatize the words
        word_tokenize_list_no_punct_lc_no_stowords_lemmatized = [lemmatize_word(x) for x in word_tokenized_no_punct_no_sw]
        
        # 4th snowball stemmer for Italian
        #word_tokenize_list_no_punct_lc_no_stowords_stem = [ita_stemmer.stem(i) for i in word_tokenized_no_punct_no_sw]
        a=' '.join(word_tokenize_list_no_punct_lc_no_stowords_lemmatized)
        if (a==''):
            return np.NaN
        else:
            return a

np.random.seed(123)

#Preprocessing
data_sal=pd.read_csv('../Data/salvini_tweets.csv',header=None,names=['no','date','text','tag'])
data_dim=pd.read_csv('../Data/dimaio_tweets.csv',header=None,names=['no','date','text','tag'])
data_ren=pd.read_csv('../Data/renzi_tweets.csv',header=None,names=['no','date','text','tag'])

data_sal['cand']='Salvini'
data_dim['cand']='Di Maio'
data_ren['cand']='Renzi'

data = pd.concat([data_sal,data_dim,data_ren])
data = data.sample(frac=1).reset_index(drop=True)
data = data[['text','cand']]

nltk.download('stopwords')
nltk.download('punkt')

# dictionary of Italian stop-words
it_stop_words = nltk.corpus.stopwords.words('italian')
# Snowball stemmer with rules for the Italian language
ita_stemmer = nltk.stem.snowball.ItalianStemmer()

#Cleaning and Stemming
data['text']=data['text'].apply(cleaner)
data=data.dropna().reset_index(drop=True)

# Count
print(data.groupby(data['cand']).count())

# Total vocabulary
#print(data['text'].apply(lambda x: len(x.split(' '))).sum())

# Unique vocabs
vocab = set()
data['text'].str.split().apply(vocab.update)
#print(len(vocab))

# Calculate frequencies for each politician
freq = {}
freq['Di Maio'] = {}
freq['Renzi'] = {}
freq['Salvini'] = {}
for idx, line in data.iterrows():
    pol = line['cand']
    for word in line['text'].split():
        if word not in freq[pol].keys():
            freq[pol][word] = 1
        else:
            freq[pol][word] = freq[pol][word] + 1
            
size = 350

data_sal=data.loc[data['cand'] == 'Salvini'].head(size)
data_dim=data.loc[data['cand'] == 'Di Maio'].head(size)
data_ren=data.loc[data['cand'] == 'Renzi'].head(size)
data=pd.concat([data_sal,data_dim,data_ren])
data = data.sample(frac=1).reset_index(drop=True)
data=data[['text','cand']]

x=data.text
y=data.cand
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=1)

print('\nx_train shape: '+ str(x_train.shape))
print('y_train shape: '+ str(y_train.shape))
print('x_test shape: '+ str(x_test.shape))
print('y_test shape: '+ str(y_test.shape))

#print(y_train.groupby(y_train.iloc[:,]).count())
#print(y_test.groupby(y_test.iloc[:,]).count())

# Tf-idf + Logistic Regression
tt=TfidfVectorizer(ngram_range=(1,2),max_features=20000)
tt.fit(x_train)
x_train_idf=tt.transform(x_train)
x_test_idf=tt.transform(x_test)
lr=LogisticRegression(penalty='l2',C=10)
lr.fit(x_train_idf,y_train)
preds_train=lr.predict(x_train_idf)
print("\nAccuracy of TF-idf + Logistic Regression on the Training Set:", accuracy_score(y_pred=preds_train,y_true=y_train))
print("F1-score of TF-idf + Logistic Regression on the Training Set:", f1_score(y_pred=preds_train,y_true=y_train, average='weighted'))
preds_test=lr.predict(x_test_idf)
print("Accuracy of TF-idf + Logistic Regression on the Test Set:", accuracy_score(y_pred=preds_test,y_true=y_test))
print("F1-score of TF-idf + Logistic Regression on the Test Set:", f1_score(y_pred=preds_test,y_true=y_test, average='weighted'))
#confusion_matrix(y_pred=preds_test,y_true=y_test)

# Tf-idf + Suppport Vector Machines (SVM)
tt4=TfidfVectorizer(ngram_range=(1,2),max_features=20000)
tt4.fit(x_train)
x_train_idf=tt4.transform(x_train)
x_test_idf=tt4.transform(x_test)

svc=SVC(kernel='linear', C=5)
svc.fit(x_train_idf,y_train)
preds_train=svc.predict(x_train_idf)
print("\nAccuracy of TF-idf + SVM on the Training Set:",accuracy_score(y_pred=preds_train,y_true=y_train))
print("F1-score of TF-idf + SVM on the Training Set:",f1_score(y_pred=preds_train,y_true=y_train, average='weighted'))
preds_test=svc.predict(x_test_idf)
print("Accuracy of TF-idf + SVM on the Test Set:",accuracy_score(y_pred=preds_test,y_true=y_test))
print("F1-score of TF-idf + SVM on the Test Set:",f1_score(y_pred=preds_test,y_true=y_test, average='weighted'))