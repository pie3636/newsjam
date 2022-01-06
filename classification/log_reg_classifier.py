#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#for word embedding
import gensim
from gensim.models import Word2Vec


# In[10]:


st = '19h40 : Les hospitalisations encore en hausse, 246 décès en 24h. Selon les données de Santé publique France, 20 688 malades du Covid-19 sont actuellement hospitalisés, contre 20 186 mardi.. Parmi ces malades hospitalisés, 3 695 sont actuellement pris en charge dans les services de soins critiques, contre 3 665 la veille 246 décès liés au Covid-19 ont été enregistrés à l\'hôpital ces dernières 24 heures.'


# In[11]:


def preprocess(text):
    """this function deletes extra characters,
    converts strings to lowercase, and removes punctuation"""
    
    text = text.lower() 
    text = text.strip()  
    text = re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text


# In[12]:


def stopword(string):
    """this function removes french stopwords"""
    
    a = [i for i in string.split() if i not in stopwords.words('french')]
    return ' '.join(a)


# In[13]:


wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def tokenize(string):
    return nltk.word_tokenize(string)


# In[14]:


def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))


# In[32]:


proc = finalpreprocess(st)


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn import metrics


# In[28]:


data = pd.read_csv('preprocessed_data.csv')


# In[49]:


tf_idf = Pipeline([('cv',CountVectorizer()), ('tfidf_transformer',TfidfTransformer(smooth_idf=True,use_idf=True))])

x_train = data['clean_text']
x_train_CV  = tf_idf.fit_transform(x_train)


# In[50]:


scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',
                                    random_state=0, C=1.0, penalty='l2',class_weight = "balanced")


# In[51]:


model = scikit_log_reg.fit(x_train_CV, data['relevance'])


# In[62]:


y_pred = model.predict(tf_idf.transform([proc]))


# In[63]:


y_pred


# In[66]:


def classifier(st):
    """function accepts strings!!!"""
    proc = finalpreprocess(st)
    vectorized = tf_idf.transform([proc])
    return model.predict(vectorized)[0]==1


# In[68]:


classifier(st)


# In[ ]:




