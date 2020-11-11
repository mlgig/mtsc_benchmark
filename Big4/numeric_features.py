import pandas as pd
import numpy as np
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.preprocessing import LabelEncoder


class numeric_features:
    """
    Find the numeric features in multivariate time series dataset.
    Input path and dataset name
    """
    def __init__(self, path, dataset_name):

        self.path = path
        self.dataset_name = dataset_name
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.train_x, self.train_y = load_from_tsfile_to_dataframe(self.path + '/' + self.dataset_name + "_TRAIN.ts")
        self.test_x, self.test_y = load_from_tsfile_to_dataframe(self.path + "/" + self.dataset_name + "_TEST.ts")

        le = LabelEncoder().fit(self.train_y)
        #self.train_y = le.transform(self.train_y)
        #self.test_y = le.transform(self.test_y) 
        #print(self.test_y)

    def create_features(self):
        """
        Creates numeric features from the dataset namely: max, min, avg
        """

        cols = ["max_" + str(i) for i in range(self.train_x.shape[1])]
        df_max_train = pd.DataFrame(columns=cols)
        df_max_test = pd.DataFrame(columns=df_max_train.columns)

        for i, col in enumerate(df_max_train.columns):

            df_max_train[col] = self.train_x.iloc[:, i].apply(lambda x: max(x))
            df_max_test[col] = self.test_x.iloc[:, i].apply(lambda x: max(x))

        cols = ["min_" + str(i) for i in range(self.train_x.shape[1])]
        df_min_train = pd.DataFrame(columns=cols)
        df_min_test = pd.DataFrame(columns=df_min_train.columns)

        for i, col in enumerate(df_min_train.columns):

            df_min_train[col] = self.train_x.iloc[:, i].apply(lambda x: min(x))
            df_min_test[col] = self.test_x.iloc[:, i].apply(lambda x: min(x))
        
        cols = ["avg_" + str(i) for i in range(self.train_x.shape[1])]
        df_avg_train = pd.DataFrame(columns=cols)
        df_avg_test = pd.DataFrame(columns=df_avg_train.columns)

        for i, col in enumerate(df_avg_train.columns):

            df_avg_train[col] = self.train_x.iloc[:, i].apply(lambda x: np.mean(x))
            df_avg_test[col] = self.test_x.iloc[:, i].apply(lambda x: np.mean(x))

        cols = ["std_" + str(i) for i in range(self.train_x.shape[1])]
        df_std_train = pd.DataFrame(columns=cols)
        df_std_test = pd.DataFrame(columns=df_std_train.columns)

        for i, col in enumerate(df_std_train.columns):

            df_std_train[col] = self.train_x.iloc[:, i].apply(lambda x: np.std(x))
            df_std_test[col] = self.test_x.iloc[:, i].apply(lambda x: np.std(x))

        self.df_train = pd.concat([df_max_train, df_min_train, df_avg_train, df_std_train], axis=1)
        self.df_train['label'] = pd.Series(self.train_y)
        self.df_test = pd.concat([df_max_test, df_min_test, df_avg_test, df_std_test], axis=1)
        self.df_test['label'] = pd.Series(self.test_y)
        return self.df_train, self.df_test

if __name__ == "__main__":
    obj = numeric_features('../mtsc/data/AtrialFibrillation', 'AtrialFibrillation')
    df_train, df_test = obj.create_features()
    print(df_train.head())#, df_test.columns)