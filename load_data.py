# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:55:27 2017

@author: Jason
"""
import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder
from nltk import TweetTokenizer
from nltk.corpus import stopwords

def load_data():
    with open('data_20170112.jsonl','r') as f:
        data = f.readlines()
    data = map(lambda x:x.rstrip(), data)
    data_json = "[" + ','.join(data) + "]"
    return pd.read_json(data_json)
    
def transform_data(data,tokenizer=TweetTokenizer(),
                   stopwords=stopwords.words('english')): 
    le = LabelEncoder()
    y = le.fit_transform(data.sent)
    X = data.text.apply(prepare_text,tokenizer=tokenizer,stopwords=stopwords)
    return X,y
    
    
def prepare_text(text,tokenizer = TweetTokenizer(),stopwords=stopwords):
    text = re.sub(r'\$\w*|\&[#\w]*\;','',text).lower()
    tokens = tokenizer.tokenize(text)
    return [tok for tok in tokens if tok not in stopwords and not tok.isdigit()]
    
    
if __name__=='__main__':
    data = load_data()
    punct = list(string.punctuation)
    stopwords_list = stopwords.words('english') + punct + ['rt','via','...']
    X,y = transform_data(data,stopwords=stopwords_list)