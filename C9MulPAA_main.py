# Created by bhaskar at 19/03/2020

import click
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd())
import pandas as pd
from multiprocessing import Process
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from C9MulPAA.features import PAAStat
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
import time


def agent(path, dataset, seg, folder, paa=True):

    start = time.time()
    train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TRAIN.ts")
    test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TEST.ts")

    print(f"{dataset}: Train Shape {train_x.shape}")
    print(f"{dataset}: Test Shape {test_x.shape}")

    model = Pipeline([
        ('data_transform', PAAStat(paa_=paa, seg_=seg)),
        ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                    normalize=True, class_weight='balanced'))

    ])

    model.fit(train_x.values, train_y)
    preds = model.predict(test_x.values)
    acc1 = accuracy_score(preds, test_y) * 100

    end = time.time()

    results = pd.DataFrame({'Dataset': dataset, 'AccuracyRidge': [acc1], 'Time': [end - start]})
    print(results)
    temp_path = './'+folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    results.to_csv(os.path.join(temp_path + f'/{dataset}.csv'), index=False)


@click.command()
@click.option('--path', help="Path of datasets", required=True, type=click.Path(exists=True))
@click.option('--paa', help="PAA", type=click.Choice(['True', 'False'], case_sensitive=True))
@click.option('--folder', help="Folder to store result", required=True)
@click.option('--seg', help="compression ratio", required=True, type=float, nargs=3)
def cli(path, paa, folder, seg):

    datasets = os.walk(path)
    dataset_name = sorted([dataset[1] for dataset in datasets if len(dataset[1]) > 1][0])
    print(dataset_name)
    #dataset_name = ['InsectWingbeat']
    seg = list(seg)
    processes = []
    for dataset in dataset_name:
        proc = Process(target=agent, args=(path, dataset, seg, folder, paa))
        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    cli()
