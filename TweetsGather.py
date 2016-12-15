#Import the necessary methods from tweepy library
import csv
import json

from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
from tweepy.streaming import StreamListener


#Variables that contains the user credentials to access Twitter API 
access_token = "1954399566-lwKXMZEJiyFuVtdMvWKSr3kpMT80tIAiXKasazN"
access_token_secret = "dQgwsZPXv4fAzkviW2hPjRpDSnoIdQPFitTxACKZfAIIr"
consumer_key = "Fm0VMlHqFrIZjbfTQDTuCopWK"
consumer_secret = "lSZoSFY48FAaBkdZ478lsQBpVA2z8FzFAKeerMLTl7aBauX463"

if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    #hashtag for movies
    query = '#storks'
    max_tweets = 10000
    
    csv_out = open("\\tweets_out_#storks.csv", mode='wb') #opens csv file
    f = open('\\jsonData#storks.json','wb')
    writer = csv.writer(csv_out) #create the csv writer object
    fields = ['created_at','text'] #field names
    writer.writerow(fields) #writes field
    
    tweets = []
    last_id = -1
    error = "no error"
    while len(tweets) < max_tweets:
        count = max_tweets - len(tweets)
        try:
            new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1), lang="en")
            if not new_tweets:
                error = "no new tweets"
                break
            print len(tweets)
            print "\n"
            tweets.extend(new_tweets)
            last_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # depending on TweepError.code, one may want to retry or wait
            # to keep things simple, we will give up on an error
            print e
            error = "error occurred"
            break
        
    for status in tweets: 
        if not hasattr(status, 'retweeted_status'):
            writer.writerow([status.created_at,status.text.encode('unicode_escape')])
            f.write(json.dumps(status._json))            
            
    print "\n\nLast ID: "
    print last_id
    print "\n\nTweets: "
    print len(tweets)
    print error
    f.close()
