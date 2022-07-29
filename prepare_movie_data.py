import sys
sys.path.insert(0, '../lib')  # noqa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class User:
    def __init__(self, id):
        self.id = id
        self.positive = []
        self.negative = []

    def add_positive(self, movie_id):
        self.positive.append(movie_id)

    def add_negative(self, movie_id):
        self.negative.append(movie_id)

    def get_positive(self):
        return self.positive

    def get_negative(self):
        return self.negative



class EventsGenerator:
    NUM_OF_OPENED_MOVIES_PER_USER = 20
    NUM_OF_USERS = 1000

    def __init__(self, learning_data, buy_probability):
        self.learning_data = learning_data
        self.buy_probability = buy_probability
        self.users = []
        for id in range(1, self.NUM_OF_USERS):
            self.users.append(User(id))

    def run(self, pairwise=False):
        for user in self.users:
            opened_movies = np.random.choice(self.learning_data.index.values, self.NUM_OF_OPENED_MOVIES_PER_USER)
            self.__add_positives_and_negatives_to(user, opened_movies)

        if pairwise:
            return self.__build_pairwise_events_data()
        else:
            return self.__build_events_data()

    def __add_positives_and_negatives_to(self, user, opened_movies):
        for movie_id in opened_movies:
            if np.random.binomial(1, self.buy_probability.loc[movie_id]):
                user.add_positive(movie_id)
            else:
                user.add_negative(movie_id)

    def __build_events_data(self):
        events_data = []

        for user in self.users:
            for positive_id in user.get_positive():
                tmp = self.learning_data.loc[positive_id].to_dict()
                tmp['outcome'] = 1
                events_data += [tmp]

            for negative_id in user.get_negative():
                tmp = self.learning_data.loc[negative_id].to_dict()
                tmp['outcome'] = 0
                events_data += [tmp]

        return pd.DataFrame(events_data)

    def __build_pairwise_events_data(self):
        events_data = []

        for i, user in enumerate(self.users):
            #print("{} of {}".format(i, len(self.users)))
            positives = user.get_positive()
            negatives = user.get_negative()

            sample_size = min(len(positives), len(negatives))

            positives = np.random.choice(positives, sample_size)
            negatives = np.random.choice(negatives, sample_size)

            # print("Adding {} events".format(str(len(positives) * len(negatives) * 2)))
            for positive in positives:
                for negative in negatives:
                    e1 = self.learning_data.loc[positive].values
                    e2 = self.learning_data.loc[negative].values

                    pos_neg_example = np.concatenate([e1, e2, [1]])
                    neg_pos_example = np.concatenate([e2, e1, [0]])

                    events_data.append(pos_neg_example)
                    events_data.append(neg_pos_example)

        c1 = [c + '_1' for c in self.learning_data.columns]
        c2 = [c + '_2' for c in self.learning_data.columns]
        return pd.DataFrame(events_data, columns=np.concatenate([c1, c2, ['outcome']]))


class MovieData:
    def __init__(self, pairwise=True):
        self.movie_data = None
        self.learning_data = None

        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

        self.create_movie_data()
        self.compute_price_and_buy_probability()
        self.build_learning_data()
        self.events_data = EventsGenerator(self.learning_data,
                                           self.movie_data['buy_probability']).run(pairwise=True)
        df = pd.DataFrame(self.get_feature_columns_from_learning_data(pairwise))
        self.feature_columns = df[0].values

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

    def build_learning_data(self):
        feature_columns = np.setdiff1d(self.movie_data.columns, np.array(['title', 'buy_probability']))
        self.learning_data = self.movie_data.loc[:, feature_columns]

        scaler = StandardScaler()
        self.learning_data.loc[:, 'price'] = scaler.fit_transform(self.learning_data[['price']])
        self.learning_data['ratings_average'] = scaler.fit_transform(self.learning_data[['ratings_average']])
        self.learning_data['ratings_count'] = scaler.fit_transform(self.learning_data[['ratings_count']])
        self.learning_data['release_date'] = self.learning_data['release_date'].apply(lambda x: x.year)
        self.learning_data['release_date'] = scaler.fit_transform(self.learning_data[['release_date']])

    def get_feature_columns_from_learning_data(self, pairwise=False):
        if not pairwise:
            return self.learning_data.columns.values
        else:
            f1 = [c + '_1' for c in self.learning_data.columns.values]
            f2 = [c + '_2' for c in self.learning_data.columns.values]
            f1.extend(f2)
            return np.asarray(f1)

    def get_train_and_validation_data(self):
        X = self.events_data.loc[:, self.feature_columns].values.astype(np.float32)
        # print('overall input shape: ' + str(X.shape))

        y = self.events_data.loc[:, ['outcome']].values.astype(np.float32).ravel()
        # print('overall output shape: ' + str(y.shape))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                              test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                         test_size=0.25,
                         )
        # print('training input shape: ' + str(self.X_train.shape))
        # print('training output shape: ' + str(self.y_train.shape))
        #
        # print('testing input shape: ' + str(self.X_val.shape))
        # print('testing output shape: ' + str(self.y_val.shape))

        return self.X_train, self.X_val, self.y_train, self.y_val, self.X_test, self.y_test


if __name__ == '__main__':
    movie_data = MovieData()
    movie_data.learning_data.head()





