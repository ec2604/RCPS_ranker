import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def generate_year_msd_data():
    # Assumes user has UCI dataset downloaded and unzipped.
    np.random.seed(1)
    df = pd.read_csv('YearPredictionMSD.txt', header=None)
    subset_idx = np.arange(len(df))
    np.random.shuffle(subset_idx)
    df = df.iloc[subset_idx[:10000]]
    rows = []
    for row in tqdm(df.iterrows()):
        rows.append(df.sample(5).values - row[1].values)
    data = np.vstack(rows)
    data[:, 0] = data[:, 0] > 0
    np.save('./YearMSD_data.npy', data)


class YearMSDData:
    def __init__(self):
        self.data = np.load('./YearMSD_data.npy')[:10000]
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

    def get_train_and_validation_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[:, 1:], self.data[:, 0],
                                                                                test_size=0.2,
                                                                                )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                              test_size=0.25,
                                                                              )
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        return self.X_train, self.X_val, self.y_train, self.y_val, self.X_test, self.y_test
