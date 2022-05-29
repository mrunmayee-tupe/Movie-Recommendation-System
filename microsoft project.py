#!/usr/bin/env python
# coding: utf-8

# In[9]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[10]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[11]:


credits.head()


# In[12]:


movies.shape


# In[13]:


movies = movies.merge(credits,on='title')


# In[14]:


import ast


# In[15]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[16]:


movies.dropna(inplace=True)


# In[17]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[18]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[19]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[20]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[21]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[22]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[23]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[24]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[25]:


#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


# In[26]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[27]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[28]:


movies.head()


# In[29]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[30]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[31]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[32]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[33]:


new['tags'] = new['tags'].apply(lambda x:x.lower())
new.head()


# In[82]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[83]:


vectors = cv.fit_transform(new['tags']).toarray()


# In[84]:


vectors[0]


# In[85]:


cv.get_feature_names()


# In[86]:


import nltk


# In[87]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[88]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[89]:


new['tags'] = new['tags'].apply(stem)


# In[90]:


from sklearn.metrics.pairwise import cosine_similarity


# In[93]:


similarity = cosine_similarity(vectors)


# In[106]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[121]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[124]:


recommend('Batman Begins')


# In[125]:


new.iloc[1216].title


# In[126]:


import pickle


# In[128]:


pickle.dump(new,open('movies_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




