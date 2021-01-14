# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:15:17 2020

@author: asus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('C:/Users/asus/Desktop/MIT/Projet tutoré/Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)
import re # Regular expressions
import nltk # Natural language tool kit
from nltk.corpus import stopwords # This will help us get rid of useless words.
# Extra needed packages
nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

stop_words = stopwords.words('english')
import pickle
filename = 'customer_satisfaction_model.sav'


loaded_model = pickle.load(open(filename, 'rb'))


aaaa = []
text = 'i like the food its very good'
review = text.lower()
review = re.sub("[^a-zA-Z]", " ", review)
review = nltk.word_tokenize(review)
review = [word for word in review if word.lower() not in stop_words]
lemma = WordNetLemmatizer()
review = [lemma.lemmatize(word) for word in review]
review  = " ".join(review)
aaaa.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_loaded= cv.fit_transform(aaaa).toarray()
