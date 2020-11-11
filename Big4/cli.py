import click
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

import numeric_features as nf

def classify(train, test):
    X, y = train.iloc[:, 0:-1], train.iloc[:, -1]
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True).fit(X,y)
    preds = clf.predict(test.iloc[:, 0:-1])

    return accuracy_score(preds, test.iloc[:, -1])
    
@click.command()
@click.option('--path', help="Path of datasets", required=True, type=click.Path(exists=True))
def cli(path):
    results = pd.DataFrame(columns=['Dataset', 'Accuracy'])
    datasets = os.walk(path)
    dataset_name = sorted([dataset[1] for dataset in datasets if len(dataset[1])>1][0])
    print(dataset_name)
    for dataset in dataset_name:
        print("Dataset: ", dataset)
        obj = nf.numeric_features(path+"/"+str(dataset), dataset)
        df_train, df_test = obj.create_features()
        accuracy = classify(df_train, df_test)
        print("Accuracy: {}\n".format(accuracy))
        results = pd.concat([results, pd.DataFrame({'Dataset': dataset, 'Accuracy': [accuracy]})])

    results.to_csv('RidgeRegession_ResultsNew.csv', index=False)


if __name__ == "__main__":
    cli()
