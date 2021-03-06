# TrendInvest
With the incredible volatility in Cryptocurrency markets, it sometimes feels impossible to make investments that are better than throwing some money into a game of Blackjack. This project aims to solve that.

## TI-Data.py
This script collects a lot of useful data on ~3000 cryptocurrencies. Using LunarCrush and Reddit APIs, it seeks to aggregate mass amounts of data from Twitter and Reddit. It will collect data up to 5 weeks ago and save it into the 'data' subdirectory. You could run this once a day or once a month but the more often you run it, the better 'TI-Predict.py' will be able to perform. Full disclaimer, with the amount of data being collected and the API request limitations, this script takes a few hours to run.

*Note: Running this data more than once a day might be dentrimental to the usefulness of TI-Predict.py and it's really not necessary. Keeping a consistent schedule is best to collect quality data (e.g. everyday at noon, every sunday at 5pm, etc.)*

## TI-Predict.py
This script will clean and balance the dataset(s) collected from 'TI-Data.py' and make a prediction as to whether a coin is bullish (expected to increase) or bearish (expected to decrease) in the following week. In the future when I'm happy with the accuracy of the model, I will save the model and create an interactive script purely for the purpose of inputting a coin name and receiving a prediction/confidence score. This model will be iteratively improved as new data becomes available. For now, I'm still aggregating data and this script will be what's used to predict cryptocurrency increasing or decreasing the following week.

## Updates
5/11/2021:
- Moved API keys to config file (add this if you plan on using this locally)
- Added function to save top ten most promising coins for the next week
- Added models to predict coins that are expected to increase 5% and 10% in the next week
- Created web app using this data (updated daily) for easy accessibility to this tool (https://trendinvestweb.herokuapp.com)

4/24/2021:
- Updated training data parsed to include Twitter mentions & sentiment.
- Updated feature extractor with new useful data points.

## Notes
I'm currently getting between 55%-60% accuracy from the data I've collected so far, which seems good (way better than Vegas) but since this is a relatively new project - I need to do way more testing, data collection, tweaking, etc. Models created from this project are far from clairvoyant and should only really be used as an additional tool in your research.However, the predictions from this model can be another useful metric in the analysis of cryptocurrencies before investing.

Something interesting I've seen is that this model is pretty good at classifying 'pump and dump' coins at the beginning of the 'pump' cycle. This needs to be explored further but this model could be a helpful tool for short-term crypto investing in smaller coins that have the potential of increasing ten or one-hundred fold (i.e. SafeMoon, DogeCoin, GarliCoin, etc.) - I personally wouldn't advise investing in these types of altcoins (I see it more as gambling than investing once you hit the DogeCoin tier of cryptocurrencies) but I understand that these cryptocurrencies can be profitable, so I included them. Using these coins also gives me a lot of interesting data to work with.

Last disclaimer: I'm just working on this project for fun, therefore, it's relatively crude and has hundreds of opportunities for improvement and tuning. I will try to make improvements as time goes on but please check the commit history and updates section regularly if you plan on using this as a tool for investing.
