import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import bernoulli
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_university_ranking_data():
    np.random.seed(1)
    df = pd.read_csv('Times World University Rankings (2011-2016).csv')
    for col in ['Total_Score', 'Inter_Outlook_Rating', 'Industry_Income_Rating', 'Total_Score', '%_Female_Students']:
        df.loc[df[col] == '-', col] = df[df[col] != '-'][col].astype(np.float).mean()
        df[col] = df[col].astype(np.float)
    df['%_Female_Students'] = df['%_Female_Students'].mean(skipna=True)
    df['Num_Students'] = df['Num_Students'].mean(skipna=True)
    df['Student/Staff_Ratio'] = df['Student/Staff_Ratio'].mean(skipna=True)
    df['%_Inter_Students'] = df['%_Inter_Students'].mean(skipna=True)
    df['country_score'] = df.groupby('Country')['Total_Score'].transform('mean')
    df.loc[df['World_Rank'].str.len() > 3, 'World_Rank'] = df[df['World_Rank'].str.len() > 3]['World_Rank'].str[:3]
    df['World_Rank'] = df['World_Rank'].astype(np.float)
    df = df.drop(['University_Name', 'Country', 'Total_Score'], axis=1)
    rows = []
    for row in tqdm(df.iterrows()):
        rows.append(df[df['Year'] == row[1].Year].sample(10).values - row[1].values)
    data = np.vstack(rows)
    data[:, 0] = data[:, 0] > 0
    np.save('./University_Rankings.npy', data)

class UniversityRankingData:
    def __init__(self):
        self.data = np.load('./University_Rankings.npy')
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

    def get_train_and_validation_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[:, 1:], self.data[:, 0],
                                                                              test_size=0.2,
                                                                              )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,  self.y_train,
                                                                                test_size=0.25,
                                                                                )
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        return self.X_train, self.X_val, self.y_train, self.y_val, self.X_test, self.y_test