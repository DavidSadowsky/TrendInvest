import praw
import nltk
import datetime
import time
import calendar
import urllib.request
import ssl
import json
import requests
import csv
import pandas
from nltk.sentiment.vader import SentimentIntensityAnalyzer


LUNAR_CRUSH_API_KEY = '6k44795xkr6e23ltalsndt'

# Initialize reddit instance
reddit = praw.Reddit(client_id='24beT_d9Hxp_rQ', client_secret='8Xz6TrYt2iR-6WrUXFGHgP8K3TuSiw', user_agent='TI-Analyzer')

def utc2tolocal(utc: float):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(utc))

def get_num_reddit_posts(subreddits: list, symbol: str, name: str, start: float, end: float):
    post_count = 0
    for sr in subreddits:
        for submission in reddit.subreddit(sr).search(query=name, sort='top', limit = 1000, time_filter = 'month'):
            if submission.created_utc > start and submission.created_utc < end:
                post_count += 1
    return post_count

def get_sentiment(subreddits: list, symbol: str, name: str, start: float, end: float):
    post_count = 0
    sid = SentimentIntensityAnalyzer()
    submissions = []
    title_sentiments = []
    body_sentiments = []
    for sr in subreddits:
        for submission in reddit.subreddit(sr).search(query=name, sort='top', limit = 1000, time_filter = 'month'):
            if submission.created_utc > start and submission.created_utc < end:
                        s_title = sid.polarity_scores(submission.title)
                        s_body = sid.polarity_scores(submission.selftext)
                        title_sentiments.append(s_title['compound'])
                        body_sentiments.append(s_body['compound'])

    if len(title_sentiments) + len(body_sentiments) == 0:
        return 0 
    
    return (sum(title_sentiments) + sum(body_sentiments)) / (len(title_sentiments) + len(body_sentiments))

def get_coin_meta_info():
    url = 'https://api.lunarcrush.com/v2?data=meta&key=' + LUNAR_CRUSH_API_KEY
    coins = json.loads(urllib.request.urlopen(url).read())
    metadata = []
    for coin in coins['data']:
        metadata.append((coin['name'], coin['symbol']))
    return metadata

def get_social_feed_info(symbol: str, name: str, start: float, end: float):
    url = 'https://api.lunarcrush.com/v2?data=assets&key=' + LUNAR_CRUSH_API_KEY + '&symbol=' + symbol + '&data_points=7&interval=day&start=' + str(int(start)) + '&end=' + str(int(end))
    assets = json.loads(urllib.request.urlopen(url).read())
    data = assets['data']
    percent_change_7d = data[0]['percent_change_7d']
    num_tweets = 0
    num_reddit_posts = 0
    price = 0
    r_sentiment = 0
    t_sentiment = 0
    t_sentiment_collected = 0

    for i,info in enumerate(data[0]['timeSeries']):
        if i == 6:
            try:
                price = info['close']
            except:
                price = 0
        try:
            num_reddit_posts += int(info['reddit_posts'])
        except:
            continue
        try:
            num_tweets += int(info['tweets'])
        except:
            continue
        try:
            t_sentiment += info['average_sentiment']
            t_sentiment_collected += 1
        except:
            continue
    
    if num_reddit_posts == 0:
        num_reddit_posts = get_num_reddit_posts(['CryptoCurrency'], symbol, name, start, end)
    
    if num_reddit_posts > 0:
        r_sentiment = get_sentiment(['CryptoCurrency'], symbol, name, start, end)

    if t_sentiment_collected > 0:
        t_sentiment /= (t_sentiment_collected * 5)

    return {
        'symbol': symbol,
        'num_tweets': num_tweets,
        'num_reddit_posts': num_reddit_posts,
        'percent_change_7d': percent_change_7d,
        'r_sentiment': r_sentiment,
        't_sentiment': t_sentiment,
        'price': price
    }

def strictly_decreasing(L: list):
    return all(x>y for x, y in zip(L, L[1:]))

if __name__ == '__main__':
    # Date ranges
    today = datetime.datetime.now().timestamp()
    week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).timestamp()
    two_weeks_ago = (datetime.datetime.now() - datetime.timedelta(days=14)).timestamp()
    three_weeks_ago = (datetime.datetime.now() - datetime.timedelta(days=21)).timestamp()
    four_weeks_ago = (datetime.datetime.now() - datetime.timedelta(days=28)).timestamp()
    five_weeks_ago = (datetime.datetime.now() - datetime.timedelta(days=35)).timestamp()

    # Collect cryptocurrency symbols and tokens
    coin_data = get_coin_meta_info()

    # Storage for parsed data
    coin_names = []
    coin_symbols = []
    r_mentions_this_week = []
    r_mentions_last_week = []
    r_mentions_two_weeks_ago = []
    r_mentions_three_weeks_ago = []
    r_mentions_four_weeks_ago = []
    r_sentiment_this_week = []
    r_sentiment_last_week = []
    r_sentiment_two_weeks_ago = []
    r_sentiment_three_weeks_ago = []
    r_sentiment_four_weeks_ago = []
    t_mentions_this_week = []
    t_mentions_last_week = []
    t_mentions_two_weeks_ago = []
    t_mentions_three_weeks_ago = []
    t_mentions_four_weeks_ago = []
    t_sentiment_this_week = []
    t_sentiment_last_week = []
    t_sentiment_two_weeks_ago = []
    t_sentiment_three_weeks_ago = []
    t_sentiment_four_weeks_ago = []
    price_this_week = []
    price_last_week = []
    price_two_weeks_ago = []
    price_three_weeks_ago = []
    price_four_weeks_ago = []
    percent_change_this_week = []

    total_analyzed = 0
    for i,coin in enumerate(coin_data):
        print(i, ' | Analyzing ', coin, ' ...')
        
        # If data can't be collected for the past two weeks, discard it
        num_mentions_this_week = get_social_feed_info(coin[1], coin[0], week_ago, today)
        if num_mentions_this_week['num_reddit_posts'] == 0:
            continue

        num_mentions_one_week_ago = get_social_feed_info(coin[1], coin[0], two_weeks_ago, week_ago)
        if num_mentions_one_week_ago['num_reddit_posts'] == 0:
            continue

        num_mentions_two_weeks_ago = get_social_feed_info(coin[1], coin[0], three_weeks_ago, two_weeks_ago)
        num_mentions_three_weeks_ago = get_social_feed_info(coin[1], coin[0], four_weeks_ago, three_weeks_ago)
        num_mentions_four_weeks_ago = get_social_feed_info(coin[1], coin[0], five_weeks_ago, four_weeks_ago)
 
        
        # Store extracted features
        coin_names.append(coin[0])
        coin_symbols.append(coin[1])

        r_mentions_this_week.append(num_mentions_this_week['num_reddit_posts'])
        r_mentions_last_week.append(num_mentions_one_week_ago['num_reddit_posts'])
        r_mentions_two_weeks_ago.append(num_mentions_two_weeks_ago['num_reddit_posts'])
        r_mentions_three_weeks_ago.append(num_mentions_three_weeks_ago['num_reddit_posts'])
        r_mentions_four_weeks_ago.append(num_mentions_four_weeks_ago['num_reddit_posts'])

        r_sentiment_this_week.append(num_mentions_this_week['r_sentiment'])
        r_sentiment_last_week.append(num_mentions_one_week_ago['r_sentiment'])
        r_sentiment_two_weeks_ago.append(num_mentions_two_weeks_ago['r_sentiment'])
        r_sentiment_three_weeks_ago.append(num_mentions_three_weeks_ago['r_sentiment'])
        r_sentiment_four_weeks_ago.append(num_mentions_four_weeks_ago['r_sentiment'])

        t_mentions_this_week.append(num_mentions_this_week['num_tweets'])
        t_mentions_last_week.append(num_mentions_one_week_ago['num_tweets'])
        t_mentions_two_weeks_ago.append(num_mentions_two_weeks_ago['num_tweets'])
        t_mentions_three_weeks_ago.append(num_mentions_three_weeks_ago['num_tweets'])
        t_mentions_four_weeks_ago.append(num_mentions_four_weeks_ago['num_tweets'])

        t_sentiment_this_week.append(num_mentions_this_week['t_sentiment'])
        t_sentiment_last_week.append(num_mentions_one_week_ago['t_sentiment'])
        t_sentiment_two_weeks_ago.append(num_mentions_two_weeks_ago['t_sentiment'])
        t_sentiment_three_weeks_ago.append(num_mentions_three_weeks_ago['t_sentiment'])
        t_sentiment_four_weeks_ago.append(num_mentions_four_weeks_ago['t_sentiment'])

        price_this_week.append(num_mentions_this_week['price'])
        price_last_week.append(num_mentions_one_week_ago['price'])
        price_two_weeks_ago.append(num_mentions_two_weeks_ago['price'])
        price_three_weeks_ago.append(num_mentions_three_weeks_ago['price'])
        price_four_weeks_ago.append(num_mentions_four_weeks_ago['price'])

        percent_change_this_week.append(num_mentions_this_week['percent_change_7d'])
        
        total_analyzed += 1
        print(coin[0], ' successfully analyzed', ' | total analyzed: ', total_analyzed)
            
    # Save data to DataFrame
    d = {'Coins': coin_names,
        'Symbols': coin_symbols,
        'Reddit mentions this week': r_mentions_this_week,
        'Reddit mentions last week': r_mentions_last_week,
        'Reddit mentions two weeks ago': r_mentions_two_weeks_ago,
        'Reddit mentions three weeks ago': r_mentions_three_weeks_ago,
        'Reddit mentions four weeks ago': r_mentions_four_weeks_ago,
        'Reddit sentiment this week': r_sentiment_this_week,
        'Reddit sentiment last week': r_sentiment_last_week,
        'Reddit sentiment two weeks ago': r_sentiment_two_weeks_ago,
        'Reddit sentiment three weeks ago': r_sentiment_three_weeks_ago,
        'Reddit sentiment four weeks ago': r_sentiment_four_weeks_ago,
        'Twitter mentions this week': t_mentions_this_week,
        'Twitter mentions last week': t_mentions_last_week,
        'Twitter mentions two weeks ago': t_mentions_two_weeks_ago,
        'Twitter mentions three weeks ago': t_mentions_three_weeks_ago,
        'Twitter mentions four weeks ago': t_mentions_four_weeks_ago,
        'Twitter sentiment this week': t_sentiment_this_week,
        'Twitter sentiment last week': t_sentiment_last_week,
        'Twitter sentiment two weeks ago': t_sentiment_two_weeks_ago,
        'Twitter sentiment three weeks ago': t_sentiment_three_weeks_ago,
        'Twitter sentiment four weeks ago': t_sentiment_four_weeks_ago,
        'Price this week': price_this_week,
        'Price last week': price_last_week,
        'Price two weeks ago': price_two_weeks_ago,
        'Price three weeks ago': price_three_weeks_ago,
        'Price four weeks ago': price_four_weeks_ago,
        'Percent change this week': percent_change_this_week
        }

    df = pandas.DataFrame(d, columns = ['Coins', 
    'Symbols',
    'Reddit mentions this week', 
    'Reddit mentions last week', 
    'Reddit mentions two weeks ago',
    'Reddit mentions three weeks ago',
    'Reddit mentions four weeks ago',
    'Reddit sentiment this week',
    'Reddit sentiment last week',
    'Reddit sentiment two weeks ago',
    'Reddit sentiment three weeks ago',
    'Reddit sentiment four weeks ago',
    'Twitter mentions this week',
    'Twitter mentions last week',
    'Twitter mentions two weeks ago',
    'Twitter mentions three weeks ago',
    'Twitter mentions four weeks ago',
    'Twitter sentiment this week',
    'Twitter sentiment last week',
    'Twitter sentiment two weeks ago',
    'Twitter sentiment three weeks ago',
    'Twitter sentiment four weeks ago',
    'Price this week',
    'Price last week',
    'Price two weeks ago',
    'Price three weeks ago',
    'Price four weeks ago',
    'Percent change this week'
    ])

    file_url = 'data/crypto_data' + utc2tolocal(today) + '.csv'
    df.to_csv(file_url, index = False, header=True)
