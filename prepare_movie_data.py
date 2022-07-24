import sys

sys.path.insert(0, '../lib')  # noqa
import numpy as np
import pandas as pd
from datetime import datetime
import dateutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import pdb
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


class MovieData:
    def __init__(self):
        self.movie_data = None
        self.create_movie_data()
        self.compute_price_and_buy_probability()

    def create_movie_data(self):
        genres_data = pd.read_csv(
            'movielens-dataset/u.genre',
            sep='|',
            encoding="ISO-8859-1",
            header=None,
            names=['name', 'id']
        )
        movie_data_columns = np.append(
            ['movie_id', 'title', 'release_date', 'video_release_date', 'url'],
            genres_data['name'].values
        )
        self.movie_data = pd.read_csv(
            'movielens-dataset/u.item',
            sep='|',
            encoding="ISO-8859-1",
            header=None,
            names=movie_data_columns,
            index_col='movie_id'
        )
        selected_columns = np.append(['title', 'release_date'], genres_data['name'].values)
        self.movie_data = self.movie_data[selected_columns]
        self.movie_data['release_date'] = pd.to_datetime(self.movie_data['release_date'])

        ratings_data = pd.read_csv(
            'movielens-dataset/u.data',
            sep='\t',
            encoding="ISO-8859-1",
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        self.movie_data['ratings_average'] = ratings_data.groupby(['movie_id'])['rating'].mean()
        self.movie_data['ratings_count'] = ratings_data.groupby(['movie_id'])['rating'].count()

        """Remove null values"""

        self.movie_data[selected_columns].isnull().any()
        null_release_dates = self.movie_data[self.movie_data['release_date'].isnull()]
        assert null_release_dates.shape[0] == 1
        movie_data = self.movie_data.drop(null_release_dates.index.values)
        assert movie_data[selected_columns].isnull().any().any() == False

    def compute_price_and_buy_probability(self):
        oldest_date = pd.to_datetime(self.movie_data['release_date']).min()
        most_recent_date = pd.to_datetime(self.movie_data['release_date']).max()
        normalised_age = (most_recent_date - pd.to_datetime(self.movie_data['release_date'])) / (
                    most_recent_date - oldest_date)
        normalised_rating = (5 - self.movie_data['ratings_average']) / (5 - 1)

        self.movie_data['price'] = np.round((1 - normalised_rating) * (1 - normalised_age) * 10)
        self.movie_data = self.movie_data[self.movie_data['price'].notnull()]  # One movie had title unknown,release data unknown, etc...
        self.movie_data['buy_probability'] = 1 - self.movie_data['price'] * 0.1  # The lower the price, the more likely I am going to buy


if __name__ == '__main__':
    movie_data = MovieData()
    print(movie_data.movie_data.head())



