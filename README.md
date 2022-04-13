# Sentiment Analysis Using Twitter API

{% hint style="info" %}
This document will explain how we use the Twitter API to get the data and use the data for Sentiment Analysis
{% endhint %}

## Install and import the Library

{% code title="Import Libraries" %}
```
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import datetime as dt

# import pycountry
import re
import string

# from wordcloud import WordCloud
import wordcloud
from PIL import Image
nltk.downloader.download('vader_lexicon')

# from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
```
{% endcode %}

## Get my API keys in Twitter

To be able to use the API, we create the twitter developer account with elevated access and get the API key, API key secret, access token, and access token secret

{% code title="My API key" %}
```
consumerKey = "CDEyjgEQ3BGXLxN1fuwEgnoIQ"
consumerSecret = "28qzXyKD2Ow2nfASDF5f3pDEEH4HYejPYHf45WrOrbt5cxE6SZ"
accessToken = "1451268525776162816-2fCNixzG6Vc9NtBH68NArRceuNQ3XJ"
accessTokenSecret = "NhWwnhe8sLSllAl5tvPBceUgueSyHwLb37bCxR85Uyc92"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)
```
{% endcode %}

After we get the API key, we create a customizable input so we are able to put the keyword or hashtag and the number of tweets after we run the code.

We use the **“Tweepy”** which is an easy-to-use Python library for accessing the Twitter API and use api.search\_tweets. For the iteration of each tweets, we use the details such as below.

| Column Name | keyword in Twitter API |
| ----------- | ---------------------- |
| created at  | tweet.created\_at      |
| text        | tweet.text             |
| source      | tweet.source           |
| name        | tweet.user.name        |
| location    | tweet.user.location    |
| verified    | tweet.user.verified    |
| description | tweet.user.description |

{% hint style="info" %}
from Twitter API, we know that the maximum timeframe of recent Tweets published in the past 7 days and 2900 tweets
{% endhint %}

{% code title="Using Twitter API" %}
```
#twitter API
def percentage(part,whole):
    return 100 * float(part)/float(whole)
keyword = input("Please enter keyword or hashtag to search: ")
noOfTweet = int(input ("Please enter how many tweets to analyze: "))
tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(noOfTweet)

# create a list of records
tweet_list = []

# iterate over each tweet and corresponding user details
for tweet in tweets:
    tweet_info = {
        'created_at': tweet.created_at,
        'text': tweet.text,
        'source': tweet.source,
        'name': tweet.user.name,
        'location': tweet.user.location,
        'verified': tweet.user.verified,
        'description': tweet.user.description
    }
    tweet_list.append(tweet_info)

# create dataframe from the extracted records
tweets_df = pd.DataFrame(tweet_list)

# display the dataframe
tweets_df.head()

#makesure no duplicates
tweets_df.drop_duplicates(inplace = True)
```
{% endcode %}

After we run the code, we should be able to put the keyword and number of tweets such as below.

The result of the 2000 #tesla in Twitter

![Customizable input](<.gitbook/assets/image (4) (1).png>)

The result of the 2000 #tesla in Twitter

![Tesla in Twitter](<.gitbook/assets/image (2) (1).png>)

## Sentiment Analysis

{% hint style="info" %}
Sentiment analysis is one of the most popular use cases for NLP (Natural Language Processing)
{% endhint %}

In this part, we use Textblob to calculate **positive**, **negative**, **neutral**, **polarity** and **compound** parameters from the text.

The polarity of a sentence refers to the amount of emotion included within it, whether it is positive, negative, or neutral. It is between -1 and 1. If the polarity is -1, the sentence is negative. If the polarity is +1, the sentence is positive. And if the polarity is 0, the statement is neutral. That is, it is neither positively nor negatively charged. Subjectivity was used to express ideas, perspectives, feelings, and emotions on social media platforms. Subjectivity refers to whether or not a sentence has emotion. Additionally, it is critical to determine whether the sentence is a fact (objective) or only an opinion (subjective). This is denoted by the 0 or 1 value. If the value is 0, it is objective; if it is 1, it is subjective.

{% code title="Sentiment Analysis" %}
```
tweets_df[["polarity", "subjectivity"]] = tweets_df["text"].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

for index, row in tweets_df["text"].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score["neg"]
    neu = score["neu"]
    pos = score["pos"]
    comp = score["compound"]
 
    if neg > pos:
        tweets_df.loc[index, "sentiment"] = "negative"
    elif pos > neg:
        tweets_df.loc[index, "sentiment"] = "positive"
    else:
        tweets_df.loc[index, "sentiment"] = "neutral"

    tweets_df.loc[index, "neg"] = neg
    tweets_df.loc[index, "neu"] = neu
    tweets_df.loc[index, "pos"] = pos
    tweets_df.loc[index, "compound"] = comp

tweets_df.head(10)
```
{% endcode %}

Result of the Sentiment Analysis

![Sentiment Analysis' Result](<.gitbook/assets/image (3) (1).png>)

{% code title="Summarize" %}
```
#Number of Tweets (Total, Positive, Negative, Neutral)
positive = 0
negative = 0
neutral = 0
positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')

print("total number: ",len(tweets_df))
print(tweets_df.groupby(['sentiment']).size())
```
{% endcode %}

The Result of the Sentiment Analysis

![](<.gitbook/assets/image (6) (1).png>)

![](<.gitbook/assets/Screen Shot 2022-03-18 at 1.11.05 AM.png>)

## Word Cloud

```
#wordcloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
    
STOPWORDS = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz","fucking","rastacake","fuck","dont","doesnt","didnt","ive", "TESLA", "tesla", "Elon", "Musk", "elon", "musk", "elonmusk"]

def remove_stopwords(txt):
    words = txt.split()
    for i, word in enumerate(words):
        if word in STOPWORDS:
            words[i] = " "
    return (" ".join(words))

output_text = []

tweets_df['text']

for word in tweets_df['text']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"  
    cleantextCAP = re.sub(expression, '', cleantextprep) 
    cleantext = cleantextCAP.lower()  
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)
    
output_text   

text = ' '.join(output_text)

from wordcloud import WordCloud, STOPWORDS,  ImageColorGenerator

mask = np.array(Image.open("elon_tw.png"))
wordcloud = WordCloud(random_state=1, background_color='white', mask=mask,  stopwords = STOPWORDS).generate(text)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  
```

![](<.gitbook/assets/image (19).png>)

![](.gitbook/assets/image.png)

Word cloud output

![](<.gitbook/assets/image (5).png>)

# Sentiment Analysis Using Reddit API

### Reddit API with pushshift.io

Reddit has removed the Cloudsearch function so that the time-based search function of Reddit API does not work. Pushshift.io is an open-source API run by Jason Baumgartner. He stored Reddit past data into the server provided with an API.

#### List of endpoints

* /reddit/submission/search
* /reddit/comment/search

#### API Parameters

| Parameter  | Description                                           | Default             | Accepted Values                                           |
| ---------- | ----------------------------------------------------- | ------------------- | --------------------------------------------------------- |
| q          | Search term.                                          | N/A                 | String / Quoted String for phrases                        |
| ids        | Get specific comments via their ids                   | N/A                 | Comma-delimited base36 ids                                |
| size       | Number of results to return                           | 25                  | Integer <= 500                                            |
| fields     | One return specific fields (comma delimited)          | All Fields Returned | string or comma-delimited string                          |
| sort       | Sort results in a specific order                      | "desc"              | "asc", "desc"                                             |
| sort\_type | Sort by a specific attribute                          | "created\_utc"      | "score", "num\_comments", "created\_utc"                  |
| aggs       | Return aggregation summary                            | N/A                 | \["author", "link\_id", "created\_utc", "subreddit"]      |
| author     | Restrict to a specific author                         | N/A                 | String                                                    |
| subreddit  | Restrict to a specific subreddit                      | N/A                 | String                                                    |
| after      | Return results after this date                        | N/A                 | Epoch value or Integer + "s,m,h,d" (i.e. 30d for 30 days) |
| before     | Return results before this date                       | N/A                 | Epoch value or Integer + "s,m,h,d" (i.e. 30d for 30 days) |
| frequency  | Used with the aggs parameter when set to created\_utc | N/A                 | "second", "minute", "hour", "day"                         |
| metadata   | display metadata about the query                      | false               | "true", "false"                                           |

### GET request with Postman

To have a better understanding of the data structure of the API, a GET request is performed using Postman.

![GET Request with Postman](<.gitbook/assets/image (4).png>)

Desired parameters are as followed:&#x20;

q = Tesla

after = 2021 quarter one timestamp in epoch format

before = 2021 quarter four timestamp in epoch format

sort\_type = score

sort = desc

### Response from the server from Postman

![JSON response](<.gitbook/assets/image (15).png>)

### Reddit API using Python

#### Essential modules for Pushshift.io API

Firstly, import requests, json, pandas, and datetime modules.

```
import requests
import json
import pandas as pd
from datetime import datetime
```

#### Timestamp during Q1-Q4 in 2021

Define timestamps of the API parameters where;

**data from** and **data until** is a list of timestamp.

**size** for the maximum size of retrieving data.

**q** for user input desired keyword.

```
#epoch timestamp
data_from  = ['1609459200', '1617235200', '1625097600','1633046400']
data_until = ['1617148800', '1625011200' ,'1632960000', '1640908800']
size = 500
q = str(input())
```

![Input keyword](<.gitbook/assets/image (12).png>)

#### Create a list to store Q1-Q4 submissions & comments API URLs

```
urls_sms = []
urls_cm  = []
for i in range(len(data_from)):
    urls_sms.append(f'https://api.pushshift.io/reddit/submission/search/?{size}&sort=desc&sort_type=score&before={data_until[i]}&q={q}&after={data_from[i]}')
for i in range(len(data_from)):
    urls_cm.append(f'https://api.pushshift.io/reddit/search/comment/?{size}&q={q}&before={data_until[i]}&sort_type=score&sort=desc&after={data_from[i]}')    
```

#### Send GET Requests and store in lists

```
r_sms = []
r_cms =[]
comments = []
contents = []
for i in range(len(urls_sms)):
    r_sms.append(requests.get(urls_sms[i]))
    contents.append(r_sms[i].content)
    r_cms.append(requests.get(urls_cm[i]))
    comments.append(r_cms[i].content)
```

#### Example output from GET request

As it can be seen in the figure below, there are irrelevant JSON format data from the API that no need to be collected. Also, data in a list are still in JSON format. Hence, JSON load shall be performed next.

![Output from a request](<.gitbook/assets/image (7).png>)

#### Decode JSON format into a dictionary

Using **JSON.loads()** function to decode JSON format data from **contents** and **comments** list

```
col = ['title','subreddit','upvote_ratio', 'score','num_comments','time']
dt_sms = [[],[],[],[]]
js_contents = []
js_comments = []
for i in range(len(contents)):
    js_contents.append(json.loads(contents[i]))
    js_comments.append(json.loads(comments[i]))
```

After that, extract only useful information from the dictionary to create dataframes.

Output from **submissions** and **comments** request are different so there are two different iterations.&#x20;

#### For submissions

```
for i in range(len(js_contents)):
    for content in js_contents[i]['data']:
        dt_sms[i].append([content['title'],content['subreddit'],content['upvote_ratio'],content['score'], content['num_comments'], datetime.utcfromtimestamp(content['created_utc']).strftime('%Y-%m-%d %H:%M:%S')])
```

#### For comments

```
col_cm = ['comment', 'subreddit', 'score', 'time']
dt_cm = [[],[],[],[]]
for i in range(len(js_comments)):
    for content in js_comments[i]['data']:
        dt_cm[i].append([content['body'],content['subreddit'],content['score'], datetime.utcfromtimestamp(content['created_utc']).strftime('%Y-%m-%d %H:%M:%S')])
```

### Create dataframes

First set of dataframe for submission from Q1-Q4

```
df_sms_q1 = pd.DataFrame(dt_sms[0], columns=col)
df_sms_q2 = pd.DataFrame(dt_sms[1], columns=col)
df_sms_q3 = pd.DataFrame(dt_sms[2], columns=col)
df_sms_q4 = pd.DataFrame(dt_sms[3], columns=col)
```

Second for comments

```
df_cm_q1 = pd.DataFrame(dt_cm[0], columns=col_cm)
df_cm_q2 = pd.DataFrame(dt_cm[1], columns=col_cm)
df_cm_q3 = pd.DataFrame(dt_cm[2], columns=col_cm)
df_cm_q4 = pd.DataFrame(dt_cm[3], columns=col_cm)
```

#### Dataframe output

![](<.gitbook/assets/image (9).png>)

![](<.gitbook/assets/image (3).png>)

#### Comment dataframe

![](<.gitbook/assets/image (1).png>)

### Perform sentiment analysis with NLTK Vader

For Sentiment Analysis, VADER Sentiment Analysis from NLTK is selected where VADER is a pre-built lexicon and rule-based feeling analysis instrument that is explicitly sensitive to suppositions communicated in web-based media.

#### Import SentimentIntensityAnalyzer

Create lists for collecting sentiment scores for contents and comments.

```
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
sia_contents = [[],[],[],[]]
sia_comments = [[],[],[],[]]
```

Example of words and scores from Lexicon

![](<.gitbook/assets/image (13).png>)

#### Update lexicon

As we are doing a finding on Tesla, more keywords should be added to obtain more precise sentiment scores.

```
new_words = {
    'quarter 3': 1.5,
    'Tesla': 1.3,
    'Buy': 1.4,
    'Long': 1.2,
    'growth': 1.4,
    'increase': 1.5,
    'skyrocket': 1.3,
    'Sell': -1.4,
    'Short': -1.3,
    'bad': -1.5,
    'decrease': -1.2,
    'unwealthy': -1.4,
    'model 3/Y': 1.1,
    'model S/X': 1.1,
    'deliveries': 1.0,
    'super': 0.8,
    'helpful': 0.9,
    'bought it': 1.2,
    'happy': 1.1,
    'propulsion': 1.0,
    'new car': 1.3,
    'turn off': -0.9,
    'bad': -1.2,
    'sad': -1.4,
    'fake': -1.0,
    'expensive': -1.3 
}
sia.lexicon.update(new_words)
```

#### Sentiment Intensity Analyzer module: How is the score calculated?

Hutto & Gilbert (2016) presented a valence aware dictionary for sentiment reasoning (VADER). They constructed a list of lexicon by combining available benchmark dictionaries such as Linguistic inquiries and Word Count (LIWC) with the modern sentiment features to be able to analyse social media texts as follows (Tausczik & Pennebaker, 2010) :&#x20;

Western-style emoticons such as "XD", ":p", ":-)"

Sentiment related acronyms such as "LOL", "LMAO", "ROFL"

Commonly use slang with sentiments such as "nah", "meh", "giggly"

They define a scale of a score from -4 to 4. The scores are collected with the wisdom of the crowd approach from Amazon Mechanical Terk, an online minor tasks marketplace, with more than 90000 human raters.

#### Calculating a score from an input sentence

VADER used 5 rules, called Heuristics, to deal with the perceived intensity sentiment of the sentence-level text.

1. **Punctuation -** "Today's weather is hot!" is more intense than "Today's weather is hot".
2. **Capitalization -** "Today's weather is HOT" is more intense than "Today's weather is hot".
3. **Degree modifiers-**  Degree modifiers change the intensity of the sentence.                      "Today's weather is extremely hot.” is more intense than “Today's weather is hot.”             "Today's weather is slightly hot.” reduces the intensity.
4. **Polarity shifts due to Conjunctions-** contrast conjunctions such as "but" invert the polarity of the sentence. “Today's weather is hot, but it is bearable.” becomes mix sentiment.
5. **Catching Polarity Negation-** “Today's weather isn't really that hot.”

**The function for score calculating**

polarity\_scores function returns a dictionary of sentiment scores which contains negative, positive, neutral and compound. A compound score is calculated from a summation of valence scores of each word in the lexicon, then adjusted scores according to the heuristics, and then normalized to a new scale which is between -1 to 1. This is the most useful metric for a single unidimensional measure of a given sentence's sentiment.&#x20;

Firstly, initialize the **"sentitext"** class with text arguments of the **"polarity\_scores"** function. Extract valence such as emoticon of input text and intensifier from BOOSTER\_DICT.&#x20;

![Example of BOOSTER\_DICT elements](<.gitbook/assets/image (14).png>)

Then, append it to a **"sentiments"** list for **"sentiment\_valence"** function.

```
def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # text, words_and_emoticons, is_cap_diff = self.preprocess(text)
        sentitext = SentiText(
            text, self.constants.PUNC_LIST, self.constants.REGEX_REMOVE_PUNCTUATION
        )
        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for item in words_and_emoticons:
            valence = 0
            i = words_and_emoticons.index(item)
            if (
                i < len(words_and_emoticons) - 1
                and item.lower() == "kind"
                and words_and_emoticons[i + 1].lower() == "of"
            ) or item.lower() in self.constants.BOOSTER_DICT:
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        return self.score_valence(sentiments, text)
```

The "sentiment\_valence" and "but\_check" function check for others heuristics then make an adjustment to score.&#x20;

```
def sentiment_valence(self, valence, sentitext, item, i, sentiments):
    is_cap_diff = sentitext.is_cap_diff
    words_and_emoticons = sentitext.words_and_emoticons
    item_lowercase = item.lower()
    if item_lowercase in self.lexicon:
        # get the sentiment valence
        valence = self.lexicon[item_lowercase]

        # check if sentiment laden word is in ALL CAPS (while others aren't)
        if item.isupper() and is_cap_diff:
            if valence > 0:
                valence += self.constants.C_INCR
            else:
                valence -= self.constants.C_INCR

        for start_i in range(0, 3):
            if (
                i > start_i
                and words_and_emoticons[i - (start_i + 1)].lower()
                not in self.lexicon
            ):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                s = self.constants.scalar_inc_dec(
                    words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff
                )
                if start_i == 1 and s != 0:
                    s = s * 0.95
                if start_i == 2 and s != 0:
                    s = s * 0.9
                valence = valence + s
                valence = self._never_check(
                    valence, words_and_emoticons, start_i, i
                )
                if start_i == 2:
                    valence = self._idioms_check(valence, words_and_emoticons, i)

                    # future work: consider other sentiment-laden idioms
                    # other_idioms =
                    # {"back handed": -2, "blow smoke": -2, "blowing smoke": -2,
                    #  "upper hand": 1, "break a leg": 2,
                    #  "cooking with gas": 2, "in the black": 2, "in the red": -2,
                    #  "on the ball": 2,"under the weather": -2}

        valence = self._least_check(valence, words_and_emoticons, i)

    sentiments.append(valence)
    return sentiments
```

Finally, sum the valence of words from sentence, add emphasis regarding the punctuation in the sentence. Calculate positive, negative, and neutral scores. Normalize compound score then return the scores in dictionary.

```
def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(sum_s, text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = self.constants.normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = {
            "neg": round(neg, 3),
            "neu": round(neu, 3),
            "pos": round(pos, 3),
            "compound": round(compound, 4),
        }

        return sentiment_dict
```

#### Perform SIA and create result dataframes

**for submissions**

```
for title in df_sms_q1['title']:
    score = sia.polarity_scores(title)
    score['title'] = title
    sia_contents[0].append(score)
for title in df_sms_q2['title']:
    score = sia.polarity_scores(title)
    score['title'] = title
    sia_contents[1].append(score)
for title in df_sms_q3['title']:
    score = sia.polarity_scores(title)
    score['title'] = title
    sia_contents[2].append(score)
for title in df_sms_q4['title']:
    score = sia.polarity_scores(title)
    score['title'] = title
    sia_contents[3].append(score)
```

**for comments**

```
for comment in df_cm_q1['comment']:
    score = sia.polarity_scores(comment)
    score['comment'] = comment 
    sia_comments[0].append(score)
for comment in df_cm_q2['comment']:
    score = sia.polarity_scores(comment)
    score['comment'] = comment
    sia_comments[1].append(score)
for comment in df_cm_q3['comment']:
    score = sia.polarity_scores(comment)
    score['comment'] = comment
    sia_comments[2].append(score)
for comment in df_cm_q4['comment']:
    score = sia.polarity_scores(comment)
    score['comment'] = comment
    sia_comments[3].append(score)
```

#### **Create results dataframe from lists**

```
results_sms1_df = pd.DataFrame.from_records(sia_contents[0])
results_sms2_df = pd.DataFrame.from_records(sia_contents[1])
results_sms3_df = pd.DataFrame.from_records(sia_contents[2])
results_sms4_df = pd.DataFrame.from_records(sia_contents[3])

result_cm1_df = pd.DataFrame.from_records(sia_comments[0])
result_cm2_df = pd.DataFrame.from_records(sia_comments[1])
result_cm3_df = pd.DataFrame.from_records(sia_comments[2])
result_cm4_df = pd.DataFrame.from_records(sia_comments[3])
```

#### Add positive/negative label to the dataframe

```
upper_bound = 0
lower_bound = 0
results_sms1_df['label'] = 0
results_sms1_df.loc[results_sms1_df['compound'] >= upper_bound, 'label'] =  1
results_sms1_df.loc[results_sms1_df['compound'] <  lower_bound, 'label'] = -1

results_sms2_df['label'] = 0
results_sms2_df.loc[results_sms2_df['compound'] >= upper_bound, 'label'] =  1
results_sms2_df.loc[results_sms2_df['compound'] <  lower_bound, 'label'] = -1

results_sms3_df['label'] = 0
results_sms3_df.loc[results_sms2_df['compound'] >= upper_bound, 'label'] =  1
results_sms3_df.loc[results_sms2_df['compound'] <  lower_bound, 'label'] = -1

results_sms4_df['label'] = 0
results_sms4_df.loc[results_sms2_df['compound'] >= upper_bound, 'label'] =  1
results_sms4_df.loc[results_sms2_df['compound'] <  lower_bound, 'label'] = -1
```

#### Example of result dataframe&#x20;

![](<.gitbook/assets/image (17).png>)

#### From this point, the score of data can be used for further analysis.&#x20;

### Creating Wordcloud

Import essential modules for plotting a wordcloud.

```
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS,  ImageColorGenerator
import re
from collections import Counter
import numpy as np
from PIL import Image
```

Create a function to plot cloud of words.

```
def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off");
```

#### Filter out unnecessary words in STOPWORDS list

```
STOPWORDS = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz","fucking","rastacake","fuck","dont","doesnt","didnt","ive"]
```

create function to remove stopwords

```
def remove_stopwords(txt):
    words = txt.split()
    for i, word in enumerate(words):
        if word in STOPWORDS:
            words[i] = " "
    return (" ".join(words))
```

#### Cumulate all submissions and comments from every dataframes

```
output_text = []
for word in results_sms1_df['title']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"  # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep)  
    cleantext = cleantextCAP.lower() 
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)
    
for word in results_sms2_df['title']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"  # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep)  
    cleantext = cleantextCAP.lower() 
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)

for word in results_sms3_df['title']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"  # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep)  
    cleantext = cleantextCAP.lower() 
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)

for word in results_sms4_df['title']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"  # keep only letters, numbers and whitespace
    cleantextCAP = re.sub(expression, '', cleantextprep)  
    cleantext = cleantextCAP.lower() 
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)    
```

```
for word in result_cm1_df['comment']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"  
    cleantextCAP = re.sub(expression, '', cleantextprep) 
    cleantext = cleantextCAP.lower()  
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)

for word in result_cm2_df['comment']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"  
    cleantextCAP = re.sub(expression, '', cleantextprep) 
    cleantext = cleantextCAP.lower()  
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)
for word in result_cm3_df['comment']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"  
    cleantextCAP = re.sub(expression, '', cleantextprep) 
    cleantext = cleantextCAP.lower()  
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)
for word in result_cm4_df['comment']:
    cleantextprep = str(word)
    expression = "[^a-zA-Z ]"
    cleantextCAP = re.sub(expression, '', cleantextprep)  
    cleantext = cleantextCAP.lower() 
    cleantext = remove_stopwords(cleantext)
    bound = ''.join(cleantext)
    output_text.append(bound)
```

#### Convert list into a string

```
text = ' '.join(output_text)
```

![text used to generate wordcloud](<.gitbook/assets/image (11).png>)

#### Plotting wordcloud

```
mask = np.array(Image.open("elonmusk.png"))
wordcloud = WordCloud(random_state=1, background_color='white', mask=mask,  stopwords = STOPWORDS).generate(text)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
```

#### Wordcloud

![](<.gitbook/assets/image (18).png>)
