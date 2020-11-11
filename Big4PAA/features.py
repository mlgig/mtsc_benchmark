import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
from sklearn.base import BaseEstimator, TransformerMixin
from tslearn.piecewise import PiecewiseAggregateApproximation
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from catch22 import catch22_all
from sklearn.impute import SimpleImputer

def extract_stats(temp_list):
    return [np.mean(temp_list), np.max(temp_list), np.min(temp_list), np.std(temp_list)]


class PAAStat(BaseEstimator, TransformerMixin):

    def __init__(self, paa_, seg_):
        self.paa = paa_
        self.segs_ = [seg_]

    def fit(self, x, y=None):
        return self

    # Helper function to extract stat from a segment
    def transform(self, x, y=None):

        x_new = []
        for i, time_series in enumerate(x):
            temp = []
            for j, dim in enumerate(time_series):
                if eval(self.paa):
                    paas_ = []
                    for seg in self.segs_:
                        s = int((dim.shape[0])*seg)
                        if s < 1:
                            continue
                        #print(f"Compression: {seg}")
                        paa_per_seg = PiecewiseAggregateApproximation(n_segments=s)\
                                .fit_transform(dim).flatten()
                        paas_.extend(extract_stats(paa_per_seg))

                    temp.extend(paas_)
                else:
                    temp.extend(extract_stats(dim))
            x_new.append(temp)

        x_new = np.asarray(x_new)
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x_new)
        x_new = imp_mean.transform(x_new)

        return np.asarray(x_new)


if __name__ == '__main__':
    paa = PAAStat(paa_='True', seg_= 0.75)
    train_x, train_y = load_from_tsfile_to_dataframe("../mtsc/data/LSST/LSST_TRAIN.ts")
    s = paa.transform(train_x.values)
    print(s.shape)
