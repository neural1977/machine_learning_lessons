#!/usr/bin/env python
# coding: utf-8

# ## Analysis of Tweets

# ### Ufuk Caliskan

# In[1]:


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

np.random.seed(123)

#The accuracy table for each preprocessing and classification method combination 
scores_df = pd.DataFrame(columns=['LR','NB','SVC','RF', 'DL'], index=['Count','Tf-idf','Tf-idf_Reg','W2V', 'FT'])
scores_df

#The f-score table for each preprocessing and classification method combination 
fscores_df=pd.DataFrame(columns=['LR','NB','SVC','RF','DL'],index=['Count','Tf-idf','Tf-idf_Reg','W2V','FT'])
fscores_df

data_sal=pd.read_csv('salvini_tweets.csv', header=None, names=['no','date','text','tag'])
data_con=pd.read_csv('conte_tweets.csv',header=None,names=['no','date','text','tag'])
data_dim=pd.read_csv('dimaio_tweets.csv',header=None,names=['no','date','text','tag'])
data_zin=pd.read_csv('zingaretti_tweets.csv',header=None,names=['no','date','text','tag'])
data_ren=pd.read_csv('renzi_tweets.csv',header=None,names=['no','date','text','tag'])

data_sal['cand']='Salvini'
data_con['cand']='Conte'
data_dim['cand']='Di Maio'
data_zin['cand']='Zingaretti'
data_ren['cand']='Renzi'
data=pd.concat([data_sal,data_con,data_dim,data_zin,data_ren])
data = data.sample(frac=1).reset_index(drop=True)
data=data[['text','cand']]

nltk.download('stopwords')
nltk.download('punkt')

# dictionary of Italian stop-words
it_stop_words = nltk.corpus.stopwords.words('italian')
# Snowball stemmer with rules for the Italian language
ita_stemmer = nltk.stem.snowball.ItalianStemmer()

def lemmatize_word(input_word):
    in_word = input_word#.decode('utf-8')
    # print('Something: {}'.format(in_word))
    word_it = pattern.it.parse(
        in_word, 
        tokenize=False,  
        tag=False,  
        chunk=False,  
        lemmata=True 
    )
    # print("Input: {} Output: {}".format(in_word, word_it))
    the_lemmatized_word = word_it.split()[0][0][4]
    # print("Returning: {}".format(the_lemmatized_word))
    return the_lemmatized_word


def cleaner(s):
    
    a=re.sub('#\S*','',s)
    a=re.sub('https\S*','',a)
    a = re.sub(r'[^\w\s]','',a)
    
    a=a.strip()
    #if (a=='') or ('RT ' in s) or ('@' in s):
    if (a=='') or ('RT ' in s):
        #print(a)
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
        #a=' '.join(word_tokenized_no_punct_no_sw)
        if (a==''):
            return np.NaN
        else:
            return a


data['text']=data['text'].apply(cleaner)
data=data.dropna().reset_index(drop=True)


data=pd.read_csv('data.csv',index_col=0)

data.groupby(data['cand']).count()

#total vocabulary
data['text'].apply(lambda x: len(x.split(' '))).sum()


#unique vocabs
vocab = set()
data['text'].str.split().apply(vocab.update)
len(vocab)


# In[187]:
freq = {}
freq['Conte'] = {}
freq['Di Maio'] = {}
freq['Renzi'] = {}
freq['Salvini'] = {}
freq['Zingaretti'] = {}
for idx, line in data.iterrows():
    pol = line['cand']
    for word in line['text'].split():
        if word not in freq[pol].keys():
            freq[pol][word] = 1
        else:
            freq[pol][word] = freq[pol][word] + 1
            


# In[188]:


size = 350

data_sal=data.loc[data['cand'] == 'Salvini'].head(size)
data_con=data.loc[data['cand'] == 'Conte'].head(size)
data_dim=data.loc[data['cand'] == 'Di Maio'].head(size)
data_zin=data.loc[data['cand'] == 'Zingaretti'].head(size)
data_ren=data.loc[data['cand'] == 'Renzi'].head(size)
data=pd.concat([data_sal,data_con,data_dim,data_zin,data_ren])
data = data.sample(frac=1).reset_index(drop=True)
data=data[['text','cand']]


# In[189]:


x=data.text
y=data.cand
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=1)


# In[190]:


print('x_train shape: '+ str(x_train.shape))
print('y_train shape: '+ str(y_train.shape))
print('x_test shape: '+ str(x_test.shape))
print('y_test shape: '+ str(y_test.shape))


# In[191]:


y_train.groupby(y_train.iloc[:,]).count()


# In[192]:


y_test.groupby(y_test.iloc[:,]).count()


# # TF-IDF Vec

# ## Tf-idf + Logistic Reg.

# In[193]:


#pipe=Pipeline([
#    ('tfidf_vectorizer',TfidfVectorizer()),
#    ('lrclassifier',LogisticRegression())
#])
#n_gram_options=[(1,2),(1,3),(1,4)]
#max_feature_options=[1500,2500,5000,10000,20000]
#Cs=[0.5,1,2,5,10]
#params={'tfidf_vectorizer__ngram_range':n_gram_options,
#        'tfidf_vectorizer__max_features':max_feature_options,
#        'lrclassifier__C':Cs}


# In[194]:


#gridsearch = GridSearchCV(pipe, params, verbose=1,return_train_score=True,n_jobs=3,cv=3,error_score=0).fit(x_train, y_train)


# In[195]:


#gridsearch.best_params_


# In[196]:


tt=TfidfVectorizer(ngram_range=(1,2),max_features=20000)
tt.fit(x_train)
x_train_idf=tt.transform(x_train)
x_test_idf=tt.transform(x_test)


# In[197]:


lr=LogisticRegression(penalty='l2',C=10)
lr.fit(x_train_idf,y_train)


# In[198]:


preds_train=lr.predict(x_train_idf)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[199]:


f1_score(y_pred=preds_train,y_true=y_train, average='weighted')


# In[200]:


preds_test=lr.predict(x_test_idf)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[201]:


f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[202]:


scores_df['LR']['Tf-idf']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['LR']['Tf-idf']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[203]:


confusion_matrix(y_pred=preds_test,y_true=y_test)


# In[204]:


bas={}
for b in range(0,5):
    mydict = {}
    for i in range(0, len(lr.coef_[b])):
        if lr.coef_[b][i]>1.5:
            myc = lr.coef_[b][i]
            for word,key in tt.vocabulary_.items():
                if key == i:
                    mydict[word] = myc
    bas[b]=mydict


# In[ ]:





# ## Tf-idf + SVC

# In[36]:


#pipe4=Pipeline([
#    ('tfidf_vectorizer',TfidfVectorizer()),
#    ('classifier',SVC(kernel='linear'))
#])
#n_gram_options=[(1,2),(1,3),(1,4)]
#max_feature_options=[2500,5000,10000,20000]
#c_options=[0.1,0.5,1,5,10]
#params={'tfidf_vectorizer__ngram_range':n_gram_options,
#        'tfidf_vectorizer__max_features':max_feature_options,
#        'classifier__C':c_options}


# In[37]:


#gridsearch4 = GridSearchCV(pipe4, params, verbose=1,return_train_score=True,n_jobs=3,cv=3,error_score=0).fit(x_train, y_train)


# In[38]:


#gridsearch4.best_params_


# In[39]:


tt4=TfidfVectorizer(ngram_range=(1,2),max_features=20000)
tt4.fit(x_train)
x_train_idf=tt4.transform(x_train)
x_test_idf=tt4.transform(x_test)

svc=SVC(kernel='linear',C=5)
svc.fit(x_train_idf,y_train)


# In[40]:


preds_train=svc.predict(x_train_idf)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[41]:


f1_score(y_pred=preds_train,y_true=y_train, average='weighted')


# In[42]:


preds_test=svc.predict(x_test_idf)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[43]:


f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[44]:


scores_df['SVC']['Tf-idf']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['SVC']['Tf-idf']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[45]:


scores_df


# ## Tfidf + Naive Bayes

# In[46]:


# pipe5=Pipeline([
#     ('tfidf_vectorizer',TfidfVectorizer(stop_words=ss)),
#     ('classifier',MultinomialNB())
# ])
# n_gram_options=[(1,2),(1,3),(1,4)]
# max_feature_options=[2500,5000,10000,20000]

# params={'tfidf_vectorizer__ngram_range':n_gram_options,
#         'tfidf_vectorizer__max_features':max_feature_options}


# In[47]:


#gridsearch5 = GridSearchCV(pipe5, params, verbose=1,return_train_score=True,n_jobs=3,cv=5,error_score=0).fit(x_train, y_train)


# In[48]:


#gridsearch5.best_params_


# In[49]:


tt5=TfidfVectorizer(ngram_range=(1,3),max_features=2500)
tt5.fit(x_train)
x_train_idf=tt5.transform(x_train)
x_test_idf=tt5.transform(x_test)

nb=MultinomialNB()
nb.fit(x_train_idf,y_train)


# In[50]:


preds_train=nb.predict(x_train_idf.toarray())
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[51]:


f1_score(y_pred=preds_train,y_true=y_train, average='weighted')


# In[52]:


preds_test=nb.predict(x_test_idf.toarray())
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[53]:


f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[54]:


scores_df['NB']['Tf-idf']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['NB']['Tf-idf']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[55]:


scores_df


# ## Tfidf + RF

# In[56]:


# pipe8=Pipeline([
#     ('tfidf_vectorizer',TfidfVectorizer(stop_words=ss,ngram_range=(1,3),max_features=5000)),
#     ('classifier',RandomForestClassifier())
# ])

# n_estimators =[20,100]
# min_samples_leafs=[10,50]

# params={'classifier__n_estimators':n_estimators,
#         'classifier__min_samples_leaf':min_samples_leafs}


# In[57]:


#gridsearch8=GridSearchCV(pipe8,param_grid=params,cv=3,n_jobs=3,verbose=1).fit(x_train, y_train)


# In[58]:


tt8=TfidfVectorizer(ngram_range=(1,3),max_features=5000)
tt8.fit(x_train)
x_train_idf=tt8.transform(x_train)
x_test_idf=tt8.transform(x_test)

rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=10)
rf.fit(x_train_idf,y_train)


# In[59]:


preds_train=rf.predict(x_train_idf)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[60]:


f1_score(y_pred=preds_train,y_true=y_train, average='weighted')


# In[61]:


preds_test=rf.predict(x_test_idf)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[62]:


f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[63]:


scores_df['RF']['Tf-idf']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['RF']['Tf-idf']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# # Count Vec

# ## Count Vec + Logistic Reg.

# In[64]:


# pipe2=Pipeline([
#     ('count_vectorizer',CountVectorizer(stop_words=ss)),
#     ('lrclassifier',LogisticRegression())
# ])
# n_gram_options=[(1,2),(1,3),(1,4)]
# max_feature_options=[1500,2500,5000,10000,20000]
# cs=[0.5,1,2,5,10]
# params={'count_vectorizer__ngram_range':n_gram_options,
#         'count_vectorizer__max_features':max_feature_options,
#         'lrclassifier__C':cs
#         }


# In[65]:


#gridsearch2 = GridSearchCV(pipe2, params, verbose=1,return_train_score=True,n_jobs=3,cv=5,error_score=0).fit(x_train, y_train)


# In[66]:


#gridsearch2.best_params_


# In[67]:


tt2=CountVectorizer(ngram_range=(1,2),max_features=10000)
tt2.fit(x_train)
x_train_count=tt2.transform(x_train)
x_test_count=tt2.transform(x_test)
lr=LogisticRegression(penalty='l2',C=2)
lr.fit(x_train_count,y_train)


# In[68]:


preds_train=lr.predict(x_train_count)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[69]:


f1_score(y_pred=preds_train,y_true=y_train, average='weighted')


# In[70]:


preds_test=lr.predict(x_test_count)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[71]:


f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[72]:


scores_df['LR']['Count']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['LR']['Count']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[73]:


scores_df


# In[74]:


print(classification_report(y_pred=preds_test,y_true=y_test))


# In[75]:


confusion_matrix(y_pred=preds_test,y_true=y_test)


# ## Count Vec + SVC

# In[76]:


# pipe3=Pipeline([
#     ('count_vectorizer',CountVectorizer(stop_words=ss)),
#     ('classifier',SVC(kernel='linear'))
# ])
# n_gram_options=[(1,2),(1,3),(1,4)]
# max_feature_options=[2500,5000,10000,20000]
# c_options=[0.1,0.5,1,5,10]
# params={'count_vectorizer__ngram_range':n_gram_options,
#         'count_vectorizer__max_features':max_feature_options,
#         'classifier__C':c_options}


# In[77]:


#gridsearch3 = GridSearchCV(pipe3, params, verbose=1,return_train_score=True,n_jobs=3,cv=3,error_score=0).fit(x_train, y_train)


# In[78]:


#gridsearch3.best_params_


# In[79]:


tt3=CountVectorizer(ngram_range=(1,2),max_features=10000)
tt3.fit(x_train)
x_train_count=tt3.transform(x_train)
x_test_count=tt3.transform(x_test)

svc=SVC(kernel='linear',C=0.1)
svc.fit(x_train_count,y_train)


# In[80]:


preds_train=svc.predict(x_train_count)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[81]:


f1_score(y_pred=preds_train,y_true=y_train, average='weighted')


# In[82]:


preds_test=svc.predict(x_test_count)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[83]:


f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[84]:


scores_df['SVC']['Count']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['SVC']['Count']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[85]:


scores_df


# ## Count Vec + Naive Bayes

# In[86]:


# pipe6=Pipeline([
#     ('count_vectorizer',CountVectorizer(stop_words=ss)),
#     ('classifier',MultinomialNB())
# ])
# n_gram_options=[(1,2),(1,3),(1,4)]
# max_feature_options=[2500,5000,10000,20000]

# params={'count_vectorizer__ngram_range':n_gram_options,
#         'count_vectorizer__max_features':max_feature_options}


# In[87]:


#gridsearch6 = GridSearchCV(pipe6, params, verbose=1,return_train_score=True,n_jobs=3,cv=5,error_score=0).fit(x_train, y_train)


# In[88]:


#gridsearch6.best_params_


# In[89]:


tt6=CountVectorizer(ngram_range=(1,2),max_features=10000)
tt6.fit(x_train)
x_train_count=tt6.transform(x_train)
x_test_count=tt6.transform(x_test)

nb2=MultinomialNB()
nb2.fit(x_train_count,y_train)


# In[90]:


preds_train=nb2.predict(x_train_count.toarray())
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[91]:


f1_score(y_pred=preds_train,y_true=y_train, average='weighted')


# In[92]:


preds_test=nb2.predict(x_test_count.toarray())
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[93]:


f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[94]:


scores_df['NB']['Count']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['NB']['Count']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[95]:


scores_df


# ## Count Vec. + RF

# In[96]:


# pipe9=Pipeline([
#     ('count_vectorizer',CountVectorizer(stop_words=ss)),
#     ('classifier',RandomForestClassifier())
# ])

# n_gram_options=[(1,2),(1,3)]
# max_feature_options=[2500,5000,10000]
# n_estimators =[20,100]
# min_samples_leafs=[10,50]

# params={'count_vectorizer__ngram_range':n_gram_options,
#         'count_vectorizer__max_features':max_feature_options,
#         'classifier__n_estimators':n_estimators,
#         'classifier__min_samples_leaf':min_samples_leafs}


# In[97]:


#gridsearch9=GridSearchCV(pipe9,param_grid=params,cv=3,n_jobs=3,verbose=1).fit(x_train, y_train)


# In[98]:


#gridsearch9.best_params_


# In[99]:


tt9=CountVectorizer(ngram_range=(1,3),max_features=5000)
tt9.fit(x_train)
x_train_idf=tt9.transform(x_train)
x_test_idf=tt9.transform(x_test)

rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=10)
rf.fit(x_train_idf,y_train)


# In[100]:


preds_train=rf.predict(x_train_idf)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[101]:


preds_test=rf.predict(x_test_idf)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[102]:


scores_df['RF']['Count']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['RF']['Count']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[103]:


scores_df


# # Tf-idf class normalized

# In[104]:


t_try=CountVectorizer()
t_try.fit(x_train)
x_train_try=t_try.transform(x_train)
x_test_try=t_try.transform(x_test)


# In[105]:


new_df=pd.DataFrame(x_train_try.toarray())
new_df['cand']=y_train.reset_index(drop=True)
cands=np.unique(y_test)


# In[106]:


def entropy_calculator(idm): 
    #idm=t_try.vocabulary_[word]
    
    res=[]
    
    for c in cands:
        res.append(sum(new_df[new_df['cand']==c][idm]))
    #print('res:'+str(res))
    res_p=np.array(res)/sum(res)
    #print(['res_p:'+str(res_p)])
    val=0
    for i in range(3):
        if np.log2(res_p[i])!=-np.inf:
            val=val+res_p[i]*np.log2(res_p[i])
        
    return(-val)


# In[107]:


#akes too much time
entro_dict={x:entropy_calculator(y) for x,y in t_try.vocabulary_.items()}


# In[108]:


#with open('entro_dict.txt','w') as outfile:
#    json.dump(entro_dict, outfile)


# In[109]:


#import json
#with open('entro_dict.txt', 'r') as f:
#        entro_dict = json.load(f)


# In[110]:


def NE(word):
    entro=entro_dict[word]
    max_entro=max(entro_dict.values())
    res=(max_entro-entro)/max_entro
    return(res)


# In[111]:


ne_dict={}
for word in entro_dict.keys():
    ne_dict[word]=NE(word) 


# In[112]:


x_train_try=pd.DataFrame(x_train_try.toarray())


# In[113]:


t_try2=TfidfVectorizer()
t_try2.fit(x_train)
x_train_try2=t_try2.transform(x_train)
x_test_try2=t_try2.transform(x_test)


# In[114]:


x_train_try2=pd.DataFrame(x_train_try2.toarray())
x_test_try2=pd.DataFrame(x_test_try2.toarray())


# In[115]:


#make  inverse of word-id map
id_word_map={v:k for k,v in t_try.vocabulary_.items()}


# In[116]:


ne_array=np.array([ne_dict[id_word_map[i]] for i in range(x_test_try2.shape[1]) ])
x_train_try2=x_train_try2.mul(ne_array)
x_test_try2=x_test_try2.mul(ne_array)


# ## Tfidf++ + Logistic Reg.

# In[117]:


# cs=[1,5,10,15,20,25,30]
# gridsearch7 = GridSearchCV(LogisticRegression(),param_grid={'C':cs}, verbose=1,return_train_score=True,n_jobs=3,cv=5,error_score=0).fit(x_train_tfidf_norm, y_train)


# In[118]:


#gridsearch7.best_params_


# In[119]:


lr=LogisticRegression(penalty='l2',C=30)
lr.fit(x_train_try2,y_train)


# In[120]:


preds_train=lr.predict(x_train_try2)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[121]:


preds_test=lr.predict(x_test_try2)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[122]:


scores_df['LR']['Tf-idf_Reg']=accuracy_score(y_pred=preds_test,y_true=y_test)
f1_score(y_pred=preds_test,y_true=y_test, average='weighted')
fscores_df['LR']['Tf-idf_Reg']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[123]:


scores_df


# ## Tfidf++ + Naive Bayes

# In[124]:


nb=MultinomialNB()
nb.fit(x_train_try2,y_train)


# In[125]:


preds_train=nb.predict(x_train_try2)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[126]:


preds_test=nb.predict(x_test_try2)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[127]:


scores_df['NB']['Tf-idf_Reg']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['NB']['Tf-idf_Reg']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# # TFidf++ + RF

# In[128]:


# n_estimators =[20,100]
# min_samples_leafs=[10,50]
# gridsearch9=GridSearchCV(RandomForestClassifier(),param_grid={'n_estimators':n_estimators,'min_samples_leaf':min_samples_leafs},cv=3,n_jobs=3,verbose=1).fit(x_train_try2, y_train)


# In[129]:


# gridsearch9.best_params_


# In[130]:


rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=10)
rf.fit(x_train_try2,y_train)


# In[131]:


preds_train=rf.predict(x_train_try2)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[132]:


preds_test=rf.predict(x_test_try2)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[133]:


scores_df['RF']['Tf-idf_Reg']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['RF']['Tf-idf_Reg']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')
scores_df


# # word2vec

# Word2Vec is a method to create a vector space for the words depending on their neighbors. Given a train set, it learns for every unique word in the train set the neighbors and creates with that information a vector in a vector space with a fixed length which is defined by a parameter. We use this method to create feature vectors for each unique words in our tweets. As a model we used a predefined model by … After the words are transformed to vectors for each tweet a new vector is created by taking average of already calculated word vectors. In the end, these feature vectors are used to train the classification algorithms.

# In[134]:


#from gensim.models import KeyedVectors
import gensim
#model = 'itwiki_20180420_300d.txt'
#word_vectors = KeyedVectors.load_word2vec_format(model, binary=True, unicode_errors='replace')
model = 'it/it.bin'
word_vectors = gensim.models.Word2Vec.load(model)


# In[135]:


word_vectors.wv.vocab


# In[ ]:





# In[136]:


df_train = pd.DataFrame(columns=range(0,300),index=range(0,948))


# In[137]:


def changer(inn, word_vectors):
    inn=inn.split()
    sum_vector=np.zeros(word_vectors.vector_size)
    a=0
    for token in inn:
        try:
            sum_vector += word_vectors[token]
            #print("found "+ token)
            a=a+1
        except:
            #print("not foun")
            pass
    if a==0:
        sentence_vector=0
    else:
        sentence_vector = sum_vector / a
    return(sentence_vector)


# In[138]:


x_train.reset_index(drop=True,inplace=True)


# In[139]:


#takes too much time
for r in range(x_train.shape[0]):
    inn=x_train[r]
    s_v=changer(inn, word_vectors)
    df_train.loc[r]=s_v
df_train.to_csv('w2vdf.csv')


# In[140]:


df_train.head()


# In[141]:


# to read directly
#df_train=pd.read_csv('w2vdf.csv',index_col=0)


# In[142]:


x_train_word2vec=df_train


# In[143]:


df_test = pd.DataFrame(columns=range(0,300),index=range(0,407))
x_test.reset_index(drop=True,inplace=True)


# In[144]:


for r in range(x_test.shape[0]):
    inn=x_test[r]
    s_v=changer(inn, word_vectors)
    df_test.loc[r]=s_v
x_test_word2vec=df_test


# ## Word2vec + Logistic Reg.

# In[145]:


# C=[0.5,1,5]
# gridsearch9=GridSearchCV(LogisticRegression(),param_grid={'C':C},cv=3,n_jobs=3,verbose=1).fit(x_train_word2vec, y_train)


# In[146]:


#gridsearch9.best_params_


# In[147]:


lr=LogisticRegression(penalty='l2',C=0.5)
lr.fit(x_train_word2vec,y_train)


# In[148]:


preds_train=lr.predict(x_train_word2vec)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[149]:


x_train_word2vec.shape


# In[150]:


preds_test=lr.predict(x_test_word2vec)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[151]:


scores_df['LR']['W2V']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['LR']['W2V']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[152]:


scores_df


# # word2vec + RF

# In[153]:


# n_estimators =[20,100]
# min_samples_leafs=[10,50]
# gridsearch11=GridSearchCV(RandomForestClassifier(),param_grid={'n_estimators':n_estimators,'min_samples_leaf':min_samples_leafs},cv=3,n_jobs=3,verbose=1).fit(x_train_word2vec, y_train)


# In[154]:


#gridsearch11.best_params_


# In[155]:


rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=10)
rf.fit(x_train_word2vec,y_train)


# In[156]:


preds_train=rf.predict(x_train_word2vec)
accuracy_score(y_pred=preds_train,y_true=y_train)


# In[157]:


preds_test=rf.predict(x_test_word2vec)
accuracy_score(y_pred=preds_test,y_true=y_test)


# In[158]:


scores_df['RF']['W2V']=accuracy_score(y_pred=preds_test,y_true=y_test)
fscores_df['RF']['W2V']=f1_score(y_pred=preds_test,y_true=y_test, average='weighted')


# In[159]:


scores_df


# # W2V + DL

# In[160]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
from keras import Input
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalMaxPooling1D
from keras import Model


# In[161]:


len(vocab)


# In[162]:


labels = []
labels_index = {}
i = 0
for cand in data['cand']:
    if cand not in labels_index:
        labels_index[cand] = i
        i=i+1
    labels.append(labels_index[cand])


# In[163]:


MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT = 0.2
texts = data['text'].values.tolist()

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

dat = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labs = to_categorical(np.asarray(labels))
print('Shape of data tensor:', dat.shape)
print('Shape of label tensor:', labs.shape)

# split the data into a training set and a validation set
indices = np.arange(dat.shape[0])
np.random.shuffle(indices)
dat = dat[indices]
labs = labs[indices]
nb_validation_samples = int(VALIDATION_SPLIT * dat.shape[0])#

x_train = dat[:-nb_validation_samples]
y_train = labs[:-nb_validation_samples]
x_val = dat[-nb_validation_samples:]
y_val = labs[-nb_validation_samples:]


# In[164]:


def prepare_embedding_matrix(word_index, max_words = 20000, embeddings_model = None, embedding_size = 300):
# prepare embedding matrix
    num_words = min(max_words, len(word_index))
    embedding_matrix = np.zeros((num_words+1, embedding_size))
    not_matched = 0
    for word, i in word_index.items():
        if i >= max_words:
            continue
        if word.lower() in embeddings_model.wv.vocab: # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embeddings_model[word.lower()] # gets the embedding vector associated to the word 
        else:
            not_matched = not_matched + 1 # words of training set not present in word embeddings
    return(embedding_matrix)


# In[165]:


embedding_matrix = prepare_embedding_matrix(word_index,embeddings_model=word_vectors)


# In[166]:


#embedding_layer = Embedding(input_dim = embedding_matrix.shape[0], output_dim = embedding_matrix.shape[1]
#                      , weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)


# In[167]:


#EMBEDDING_DIM = 300
#
#embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#for word, i in word_index.items():
#    if word in word_vectors.wv.vocab:
#        embedding_matrix[i] = word_vectors[word]
    #if embedding_vector is not None:
    #    # words not found in embedding index will be all-zeros.
    #    embedding_matrix[i] = embedding_vector


# In[168]:


EMBEDDING_DIM = 300
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# In[169]:


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(5, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=128)


# In[170]:


print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
#x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))


# In[171]:


print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = Dropout(0.1)(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = Dropout(0.1)(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = Dropout(0.1)(x)
x = GlobalMaxPooling1D()(x)
#x = Flatten()(x)
x = Dense(128, activation='relu')(x)

preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))


# In[172]:


#best was 52%


# In[173]:


scores_df['DL']['W2V']=0.50


# In[ ]:





# In[ ]:





# # FastText

# In[174]:


#from gensim.models.wrappers import FastText

#ft_mat = FastText.load_fasttext_format('cc.it.300.bin')


# In[221]:


import fasttext
model = fasttext.load_model("cc.it.300.bin")


# In[236]:


model.words


# In[237]:


model.labels


# In[224]:


train = pd.DataFrame()
train['text'] = x_train
train['pol'] = '__label__' + y_train


# In[225]:


train.to_csv('train.txt', sep=' ', index=False, header=False)


# In[226]:


model = fasttext.train_supervised('train.txt')


# In[227]:


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


# In[228]:


test = pd.DataFrame()
test['text'] = x_test
test['pol'] = '__label__' + y_test


# In[229]:


test.to_csv('test.txt', sep=' ', index=False, header=False)


# In[230]:


print_results(*model.test('test.txt'))


# In[231]:


preds = pd.DataFrame()
preds['prediction'] = model.predict(x_test.values.tolist())[0]
preds['probability'] = model.predict(x_test.values.tolist())[1]


# In[232]:


lst = []
for i in preds['prediction']:
    lst.append(i[0].lstrip('__label__'))


# In[233]:


preds['prediction'] = lst


# In[234]:


accuracy_score(y_pred=preds['prediction'],y_true=y_test)


# In[235]:


f1_score(y_pred=preds['prediction'],y_true=y_test, average='weighted')


# In[ ]:


len(preds['prediction'])


# In[ ]:


len(y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# # Plots

# In[205]:


scores_df=scores_df.apply(pd.to_numeric)
fscores_df=fscores_df.apply(pd.to_numeric)


# In[206]:


sns.set()
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(scores_df, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu",annot_kws={"size": 20})
plt.title('Comparison of the Methods',fontsize=20)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('comparison_heat.jpg')
plt.show()


# In[207]:


sns.set()
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(fscores_df, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu",annot_kws={"size": 20})
plt.title('Comparison of the Methods (F1)',fontsize=20)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('comparison_heatF1.jpg')
plt.show()


# The accuracy score for each preprocessing and classification combination is visualized. For our dataset, Logistic regression with TF-idf vectorization is giving the best result however 'Count Vectorization' is also giving good results with logistic regression and naive bayes.
# In general we were expecting regulerized tf-idf vectorization to beat ordinary tf-idf however the reesults were away from our expectation. Also another non- traditional method, Word2Vec, could not compete with tf-idf and word count vectorization methods. The one reason of it may be the usage of pretrained model for Word2Vec. Alternatively, it  may be not too successfull due to our dataset or in general short text data(tweets) in Turkish. 
# Between the classification methods we can see tha Random Forest is the less successfull one which was something that was already knwon to be less precise compared to Logistic Regression, SVC and Naive Bayes. Again as expected Naive Bayes works good with word count vectorization but not as good with Tf-idf. 

# # Word Cloud

# The important words(features) chosen by the most successful method combination are  visualized as word cloud in this section for each candidate.(The results are translated to English.)

# In[208]:


sorted_by_value = [sorted(bas[i].items(), key=lambda kv: kv[1],reverse=True) for i in range(0,5)]
sorted_by_value


# In[209]:


vals0=[i[1] for i in sorted_by_value[0][:17]]

vals1=[i[1] for i in sorted_by_value[1][:17]]

vals2=[i[1] for i in sorted_by_value[2][:17]]

vals3=[i[1] for i in sorted_by_value[3][:17]]

vals4=[i[1] for i in sorted_by_value[4][:17]]


# In[210]:


from wordcloud import WordCloud

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',

        max_words=30,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate_from_frequencies(data)

    fig = plt.figure(1, figsize=(12, 12))
    if title: 
        plt.title(title, fontsize=30)
    plt.axis('off')


    plt.imshow(wordcloud)
    plt.savefig('%s_wordclodud.jpg' %title)
    plt.show()
###########
show_wordcloud(bas[0],"Conte")
show_wordcloud(bas[1],"Di Maio")
show_wordcloud(bas[2],"Renzi")
show_wordcloud(bas[3],"Salvini")
show_wordcloud(bas[4],"Zingaretti")


# In[211]:


show_wordcloud(freq['Conte'], "Conte Freq")
show_wordcloud(freq["Di Maio"],"Di Maio Freq")
show_wordcloud(freq["Renzi"],"Renzi Freq")
show_wordcloud(freq["Salvini"],"Salvini Freq")
show_wordcloud(freq["Zingaretti"],"Zingaretti Freq")


# # Prediting newspapers

# In[212]:


data_cor=pd.read_csv('Corriere_tweets.csv',header=None,names=['no','date','text','tag'])
data_fog=pd.read_csv('ilfoglio_it_tweets.csv',header=None,names=['no','date','text','tag'])
data_tem=pd.read_csv('tempoweb_tweets.csv',header=None,names=['no','date','text','tag'])
data_fat=pd.read_csv('fattoquotidiano_tweets.csv',header=None,names=['no','date','text','tag'])
data_gio=pd.read_csv('ilgiornale_tweets.csv',header=None,names=['no','date','text','tag'])
data_rep=pd.read_csv('repubblica_tweets.csv',header=None,names=['no','date','text','tag'])
data_mes=pd.read_csv('ilmessaggeroit_tweets.csv',header=None,names=['no','date','text','tag'])


data_cor['name']='Corriere'
data_fog['name']='Il Foglio'
data_tem['name']='Tempo'
data_fat['name']='Fatto Quotidiano'
data_gio['name']='Il Giornale'
data_rep['name']='Repubblica'
data_mes['name']='Il Messaggero'
data_news=pd.concat([data_cor,data_fog,data_tem,data_fat,data_gio, data_rep,data_mes])
data_news = data_news.sample(frac=1).reset_index(drop=True)
data_news=data_news[['text','name']]


# In[213]:


data_news['text']=data_news['text'].apply(cleaner)
data_news=data_news.dropna().reset_index(drop=True)


# In[214]:


#data_news.to_csv('data_news.csv')
#data_news=pd.read_csv('data_news.csv',index_col=0)


# In[215]:


data_news.groupby(data_news['name']).count()


# In[216]:


size = 1000

data_cor=data_news.loc[data_news['name'] == 'Corriere'].head(size)
data_fog=data_news.loc[data_news['name'] == 'Il Foglio'].head(size)
data_tem=data_news.loc[data_news['name'] == 'Tempo'].head(size)
data_fat=data_news.loc[data_news['name'] == 'Fatto Quotidiano'].head(size)
data_gio=data_news.loc[data_news['name'] == 'Il Giornale'].head(size)
data_rep=data_news.loc[data_news['name'] == 'Repubblica'].head(size)
data_mes=data_news.loc[data_news['name'] == 'Il Messaggero'].head(size)
data_news=pd.concat([data_cor,data_fog,data_tem,data_fat,data_gio,data_rep,data_mes])
data_news = data_news.sample(frac=1).reset_index(drop=True)
data_news=data_news[['text','name']]


# In[217]:


x_news = data_news.text
y_news = data_news.name


# In[218]:


tt=TfidfVectorizer(ngram_range=(1,2),max_features=20000)
tt.fit(x_train)
#x_train_idf=tt.transform(x_train)
x_news_idf=tt.transform(x_news)
#x_test_idf=tt.transform(x_test)


# In[219]:


news_pred=lr.predict(x_news_idf)
news_preds = pd.DataFrame({'Target': y_news, 'Prediction': news_pred})


# In[220]:


pd.crosstab(news_preds.Target,news_preds.Prediction)


# In[ ]:




