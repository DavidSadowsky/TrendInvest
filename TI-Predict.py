import pandas as pd
import matplotlib.pyplot as plt
import glob, os
import random
import nltk
import math
import numpy as np
from sklearn.utils import resample

class TrendInvestingModel:
    def init(self):
        self.classifier = None
        self.df = None

    def train_model(self):
        # Read in all data available in the data directory
        self.df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "data/*.csv"))))

        # Removes dirty/incomplete data rows
        self.clean_data()

        # Tag the remaining data rows
        tagged_data = [(self.tag_data(row), self.is_incr(row['Price this week'] - row['Price last week'])) for index, row in self.df.iterrows()]

        # Split bullish and bearish samples for resampling of the smaller dataset for balancing
        bullish_samples = []
        bearish_samples = []
        for item in tagged_data:
            if item[1] == 'Bullish':
                bullish_samples.append(item)
            else:
                bearish_samples.append(item)
        
        # Balance the dataset through oversampling
        if len(bullish_samples) > len(bearish_samples) * 1.1:
            bearish_samples = resample(bearish_samples,
                                        n_samples = len(bullish_samples),
                                        random_state = 123,
                                        replace = True
                                        )
        elif len(bearish_samples) > len(bullish_samples) * 1.1:
            bullish_samples = resample(bullish_samples,
                                        n_samples = len(bearish_samples),
                                        random_state = 123,
                                        replace = True
                                        )

        # Combine the samples for a balanced dataset and shuffle
        tagged_data = bullish_samples + bearish_samples
        random.Random(77).shuffle(tagged_data)

        # Split data into training and testing sets - 90%/10%
        size = int(len(tagged_data)* 0.1)
        train_set = tagged_data[size:]
        test_set = tagged_data[:size]

        # Train Naive Bayes Classifier
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

        # Test accuracy
        print('Model accuracy:', nltk.classify.accuracy(self.classifier, test_set))
        self.classifier.show_most_informative_features()

    # Deletes all rows with incomplete or dirty data
    def clean_data(self):
        rows_to_delete = []
        for i, row in self.df.iterrows():
            for item in row:
                if item == None or str(item) == 'nan':
                    rows_to_delete.append(i)
                    break
        self.df = self.df.drop(rows_to_delete)
                    
    # Tags data with features - this will probably grow over time
    def tag_data(self, row: list):
        strictly_increasing_r_mentions = True
        strictly_increasing_r_sentiment = True
        strictly_decreasing_r_mentions = True
        strictly_decreasing_r_sentiment = True
        strictly_increasing_t_mentions = True
        strictly_increasing_t_sentiment = True
        strictly_decreasing_t_mentions = True
        strictly_decreasing_t_sentiment = True

        r_sentiment_increased_last_week = True
        r_mentions_increased_last_week = True
        t_sentiment_increased_last_week = True
        t_mentions_increased_last_week = True

        r_sentiment_increased_two_weeks_ago = True
        r_mentions_increased_two_weeks_ago = True
        t_sentiment_increased_two_weeks_ago = True
        t_mentions_increased_two_weeks_ago = True

        r_sentiment_increased_three_weeks_ago = True
        r_mentions_increased_three_weeks_ago = True
        t_sentiment_increased_three_weeks_ago = True
        t_mentions_increased_three_weeks_ago = True

        r_mentions_last_week = row['Reddit mentions last week']
        r_mentions_two_weeks_ago = row['Reddit mentions two weeks ago']
        r_mentions_three_weeks_ago = row['Reddit mentions three weeks ago']
        r_mentions_four_weeks_ago = row['Reddit mentions four weeks ago']
        t_mentions_last_week = row['Twitter mentions last week']
        t_mentions_two_weeks_ago = row['Twitter mentions two weeks ago']
        t_mentions_three_weeks_ago = row['Twitter mentions three weeks ago']
        t_mentions_four_weeks_ago = row['Twitter mentions four weeks ago']

        r_sentiment_last_week = row['Reddit sentiment last week']
        r_sentiment_two_weeks_ago = row['Reddit sentiment two weeks ago']
        r_sentiment_three_weeks_ago = row['Reddit sentiment three weeks ago']
        r_sentiment_four_weeks_ago = row['Reddit sentiment four weeks ago']
        t_sentiment_last_week = row['Twitter sentiment last week']
        t_sentiment_two_weeks_ago = row['Twitter sentiment two weeks ago']
        t_sentiment_three_weeks_ago = row['Twitter sentiment three weeks ago']
        t_sentiment_four_weeks_ago = row['Twitter sentiment four weeks ago']

        if r_mentions_last_week < r_mentions_two_weeks_ago or r_mentions_two_weeks_ago < r_mentions_three_weeks_ago or r_mentions_three_weeks_ago < r_mentions_four_weeks_ago:
            strictly_increasing_r_mentions = False
        if r_sentiment_last_week < r_sentiment_two_weeks_ago or r_sentiment_two_weeks_ago < r_sentiment_three_weeks_ago or r_sentiment_three_weeks_ago < r_sentiment_four_weeks_ago:
            strictly_increasing_r_sentiment = False
        if r_mentions_last_week < r_mentions_two_weeks_ago and r_mentions_two_weeks_ago < r_mentions_three_weeks_ago and r_mentions_three_weeks_ago < r_mentions_four_weeks_ago:
            strictly_decreasing_r_mentions = False
        if r_sentiment_last_week < r_sentiment_two_weeks_ago and r_sentiment_two_weeks_ago < r_sentiment_three_weeks_ago and r_sentiment_three_weeks_ago < r_sentiment_four_weeks_ago:
            strictly_decreasing_r_sentiment = False
        if t_mentions_last_week < t_mentions_two_weeks_ago or t_mentions_two_weeks_ago < t_mentions_three_weeks_ago or t_mentions_three_weeks_ago < t_mentions_four_weeks_ago:
            strictly_increasing_t_mentions = False
        if t_sentiment_last_week < t_sentiment_two_weeks_ago or t_sentiment_two_weeks_ago < t_sentiment_three_weeks_ago or t_sentiment_three_weeks_ago < t_sentiment_four_weeks_ago:
            strictly_increasing_t_sentiment = False
        if t_mentions_last_week < t_mentions_two_weeks_ago and t_mentions_two_weeks_ago < t_mentions_three_weeks_ago and t_mentions_three_weeks_ago < t_mentions_four_weeks_ago:
            strictly_decreasing_t_mentions = False
        if t_sentiment_last_week < t_sentiment_two_weeks_ago and t_sentiment_two_weeks_ago < t_sentiment_three_weeks_ago and t_sentiment_three_weeks_ago < t_sentiment_four_weeks_ago:
            strictly_decreasing_t_sentiment = False

        if r_sentiment_last_week < r_sentiment_two_weeks_ago:
            r_sentiment_increased_last_week = False
        if r_mentions_last_week < r_mentions_two_weeks_ago:
            r_mentions_increased_last_week = False
        if t_sentiment_last_week < t_sentiment_two_weeks_ago:
            t_sentiment_increased_last_week = False
        if t_mentions_last_week < t_mentions_two_weeks_ago:
            t_mentions_increased_last_week = False

        if r_sentiment_two_weeks_ago < r_sentiment_three_weeks_ago:
            r_sentiment_increased_two_weeks_ago = False
        if r_mentions_two_weeks_ago < r_mentions_three_weeks_ago:
            r_mentions_increased_two_weeks_ago = False
        if t_sentiment_two_weeks_ago < t_sentiment_three_weeks_ago:
            t_sentiment_increased_two_weeks_ago = False
        if t_mentions_two_weeks_ago < t_mentions_three_weeks_ago:
            t_mentions_increased_two_weeks_ago = False

        if r_sentiment_three_weeks_ago < r_sentiment_four_weeks_ago:
            r_sentiment_increased_three_weeks_ago = False
        if r_mentions_three_weeks_ago < r_mentions_four_weeks_ago:
            r_mentions_increased_three_weeks_ago = False
        if t_sentiment_three_weeks_ago < t_sentiment_four_weeks_ago:
            t_sentiment_increased_three_weeks_ago = False
        if t_mentions_three_weeks_ago < t_mentions_four_weeks_ago:
            t_mentions_increased_three_weeks_ago = False
        
        return {
            'strictly_increasing_r_mentions': strictly_increasing_r_mentions,
            'strictly_increasing_r_sentiment': strictly_increasing_r_sentiment,
            'strictly_decreasing_r_mentions': strictly_decreasing_r_mentions,
            'strictly_decreasing_r_sentiment': strictly_decreasing_r_sentiment,
            'r_sentiment_increased_last_week': r_sentiment_increased_last_week,
            'r_mentions_increased_last_week': r_mentions_increased_last_week,
            'r_sentiment_increased_two_weeks_ago': r_sentiment_increased_two_weeks_ago,
            'r_mentions_increased_two_weeks_ago': r_mentions_increased_two_weeks_ago,
            'r_sentiment_increased_three_weeks_ago': r_sentiment_increased_three_weeks_ago,
            'r_mentions_increased_three_weeks_ago': r_mentions_increased_three_weeks_ago,
            'strictly_increasing_t_mentions': strictly_increasing_t_mentions,
            'strictly_increasing_t_sentiment': strictly_increasing_t_sentiment,
            'strictly_decreasing_t_mentions': strictly_decreasing_t_mentions,
            'strictly_decreasing_t_sentiment': strictly_decreasing_t_sentiment,
            't_sentiment_increased_last_week': t_sentiment_increased_last_week,
            't_mentions_increased_last_week': t_mentions_increased_last_week,
            't_sentiment_increased_two_weeks_ago': t_sentiment_increased_two_weeks_ago,
            't_mentions_increased_two_weeks_ago': t_mentions_increased_two_weeks_ago,
            't_sentiment_increased_three_weeks_ago': t_sentiment_increased_three_weeks_ago,
            't_mentions_increased_three_weeks_ago': t_mentions_increased_three_weeks_ago
        }

    # Tags rows depending on whether the coin's price has increased or decreased this week
    def is_incr(self, diff: float):
        if diff > 0:
            return 'Bullish'
        else:
            return 'Bearish'
        
    # Predict what coins will do, pass in one data row at a time
    def predict(self, row: list):
        features = self.tag_data(row)
        tot_percent_change = 0
        prob_dist = self.classifier.prob_classify(features)
        print(prob_dist)
        # print(row['Coins'], '|', self.classifier.classify(features), '|', row['Percent change this week'])
        return tot_percent_change

if __name__ == '__main__':
    ti = TrendInvestingModel()
    ti.train_model()

    for i, row in ti.df.iterrows():
        ti.predict(row)
    