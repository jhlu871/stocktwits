# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:55:27 2017

@author: Jason
"""
import pandas as pd
import re
import os
import string
from sklearn.preprocessing import LabelEncoder
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.utils.extmath import density
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def load_data(path='data'):
    with open(os.path.join(path,'data_20170113.jsonl'),'r') as f:
        data = f.readlines()
    data = map(lambda x:x.rstrip(), data)
    data_json = "[" + ','.join(data) + "]"
    return pd.read_json(data_json)
    
def transform_data(data,tokenizer=TweetTokenizer(),
                   stopwords=stopwords.words('english')): 
    le = LabelEncoder()
    y = le.fit_transform(data.sent)
    X = data.text.apply(prepare_text)
    vec = CountVectorizer(binary=True,tokenizer=tokenizer.tokenize,
                          stop_words=stopwords)
    X = vec.fit_transform(X)
    return X,y,vec
    
    
def prepare_text(text):
    text = re.sub(r'\$\w*|\&[#\w]*\;','',text).lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
#    text = re.sub('@[^\s]+','',text)
    text = re.sub('[\s]+', ' ', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text = re.sub('\d*','',text)
    text = re.sub(r'\\\\\w*','',text)
#    text = re.sub(r'(.)\1{1,}',r'\1\1',text)
    text = text.strip('\'"')
    text = text.encode('ascii','ignore')
    return text

def train(clf,X,y,partial=True):
    if partial:
        clf.fit(X,y)
    else:
        clf.fit(X,y)
    return clf
    
def benchmark(clf,vec,X_train,y_train,X_test,y_test):
    print('_'*60)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(X_train,y_train)
    train_time = time.time() - t0
    print('training time : %0.3fs' % train_time)
    
    t0 = time.time()
    pred = clf.predict(X_test)
    test_time = time.time() - t0
    print('test time: %0.3fs' % test_time)
    
    score = accuracy_score(y_test,pred)
    print('accuracy: %0.3f' % score)
    

    if not str(clf).startswith('SVC'):
        print('dimensionality: %d' % clf.coef_.shape[1])
        print('density: %f' % density(clf.coef_))
        
        print('Top 10 keywords')
        n = 10
        feature_names = vec.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0],feature_names))
        top = zip(coefs_with_fns[:n],coefs_with_fns[:-(n+1):-1])
        for (coef_1,fn_1),(coef_2,fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
        print("classification report:")
        print(classification_report(y_test, pred))
        print("confusion matrix:")
        print(confusion_matrix(y_test, pred))

def predict(clf,vec,data):
    X = data.text.apply(prepare_text)
    X = vec.transform(X)
    return clf.predict(X)
    
if __name__=='__main__':
    data = load_data()
    punct = list(string.punctuation)
    stopwords_list = stopwords.words('english') + punct + ['rt','via','...','..']
    tk = TweetTokenizer(reduce_len=True,strip_handles=True)
    X,y,vec = transform_data(data,tokenizer=tk,stopwords=stopwords_list)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2)
#    clf = SVC(kernel='linear')
#    clf = MultinomialNB()
    clf = LogisticRegression()
    benchmark(clf,vec,X_train,y_train,X_test,y_test)
    
    with open('data_20170112.jsonl','r') as f:
        data = f.readlines()
    data = map(lambda x:x.rstrip(), data)
    data_json = "[" + ','.join(data) + "]"
    d = pd.read_json(data_json)
    y_pred = predict(clf,vec,d)
    le = LabelEncoder()
    print(confusion_matrix(le.fit_transform(d.sent),y_pred))