# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 22:10:00 2017

@author: Jason
"""
import json
import requests
import os
import sys
import time
import datetime

twit_token = os.environ['STOCKTWITS_TOKEN']
streams_url = 'https://api.stocktwits.com/api/2/streams/symbols.json'

def get_twits(syms,last_tweet):
    params = {'access_token':twit_token,
              'symbols':','.join(syms),
              'since':last_tweet}
    resp = requests.request('GET',streams_url,params=params).json()
    
    return resp
    
def process_twits_with_sentiment(twits,path='data'):
    with open(os.path.join(path,'data_%s.jsonl' % datetime.datetime.now().strftime('%Y%m%d')),
                  'a') as outfile:
        for twit in twits:
            if twit['entities']['sentiment']:
                data={"sent":twit['entities']['sentiment']['basic'],
                       "sym":twit['symbols'][0]['symbol'],
                       "text":twit['body']}
                json.dump(data,outfile)
                outfile.write('\n')


def run(syms):
    last_tweet = 0
    
    while True:
        resp = get_twits(syms,last_tweet)
        status = resp['response']['status']
        if status == 429:
            sys.stderr.write('Rate Limit Exceeded. Sleeping for 30 seconds.')
            time.sleep(30)
        elif status != 200:
            sys.stderr.write('Error: Status Code %s' %status)
        else:
            last_tweet = resp['cursor']['since']
            print(last_tweet,datetime.datetime.now())
            process_twits_with_sentiment(resp['messages'])
        time.sleep(10)       



if __name__ == '__main__':
    syms = ['SPY','XOM','JNJ','AAPL','MSFT','AMZN','BRK.B','FB','VIX','$USO']
    run(syms)