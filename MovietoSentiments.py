import csv
from textblob import TextBlob
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment 
import re

csv_out = open("\\1#test#Zootopia.csv", mode='wb')
csv_out2 = open("\\2#test#Zootopia.csv", mode='wb')
fields = ['text','pos','neu','neg','comp','subjectivity']
writer = csv.writer(csv_out)
writer2 = csv.writer(csv_out2)
writer.writerow(fields)
writer2.writerow(fields)
with open("\\tweets_out_#Zootopia.csv", 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    n=0;
    pos=0;
    ne=0;
    neu=0;
    comp=0;
    sub=0;
    for row in spamreader:
        text = row[1]
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
        blob = TextBlob(row[1])
        vs = vaderSentiment(row[1])
        blob2 = TextBlob(text)
        vs2 = vaderSentiment(text)
        #and blob.sentiment.subjectivity!=0 and blob.sentiment.subjectivity!=1
        if vs.get('compound')!=0 and blob.sentiment.subjectivity!=0:
            n += 1;
            pos += vs2.get('pos');
            ne += vs2.get('neg');
            neu += vs2.get('neu');
            comp += vs2.get('compound');
            sub += blob.sentiment.subjectivity;
            writer.writerow([row[1],vs.get('pos'),vs.get('neu'),vs.get('neg'),vs.get('compound'),blob.sentiment.subjectivity])
            writer2.writerow([text,vs2.get('pos'),vs2.get('neu'),vs2.get('neg'),vs2.get('compound'),blob2.sentiment.subjectivity])
            print text;
    writer.writerow(["##Results##",pos/n,neu/n,ne/n,comp/n,sub/n])
    writer2.writerow(["##Results##",pos/n,neu/n,ne/n,comp/n,sub/n])