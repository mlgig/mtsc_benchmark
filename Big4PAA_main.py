# Created by bhaskar at 19/03/2020
#cloud9 features
import click
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())
import pandas as pd
from multiprocessing import Process
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from Big4PAA.features import PAAStat
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
from multiprocessing import Process, current_process
import time
from sklearn.preprocessing import StandardScaler

def agent(path="./", dataset="" ,ratio =False,seg = 0.75, folder="temp"):

    current_process().name = dataset

    start1 = time.time()
    train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TRAIN.ts")
    test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TEST.ts")

    print(f"{dataset}: Train Shape {train_x.shape}")
    print(f"{dataset}: Test Shape {test_x.shape}")

    scaler = StandardScaler()

    transform_time1 = time.time()

    mod_train = PAAStat(paa_=ratio, seg_=seg).transform(train_x.values)
    mod_train = scaler.fit(mod_train).transform(mod_train)

    mod_test = PAAStat(paa_=ratio, seg_=seg).transform(test_x.values)
    mod_test = scaler.transform(mod_test)

    transform_time2 = time.time()
    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                    normalize=True)
    train_time1 = time.time()
    model.fit(mod_train, train_y)
    preds = model.predict(mod_test)
    train_time2 = time.time()

    acc1 = accuracy_score(preds, test_y) * 100

    end1 = time.time()
    print(f"Dataset: {dataset}, AccuracyRidge: {acc1}, Time taken: {(end1 - start1)/60}, "
          f"Transfrom_time: {(transform_time2-transform_time1)/60}, train_time: {(train_time2-train_time1)/60}")

    results = pd.DataFrame({'Dataset': dataset, 'AccuracyRidge': [acc1], 'Time (min)': [(end1 - start1)/60],
                            'Transfrom_time (min)': [(transform_time2-transform_time1)/60], 'train_time (min)': [(train_time2-train_time1)/60]})

    temp_path = './'+folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    results.to_csv(os.path.join(temp_path + f'/{dataset}.csv'), index=False)



@click.command()
@click.option('--path', help="Path of datasets", required=True, type=click.Path(exists=True))
@click.option('--red', help= "PAA", type=click.Choice(['True', 'False'], case_sensitive=True))
@click.option('--folder', help="Folder to store result", required=True)
@click.option('--seg', help="Factor to create TS segments PAA", default=0.75, type=click.FloatRange(0,1), show_default=True)
def cli(path, red, seg, folder):

    datasets = os.walk(path)
    dataset_name = sorted([dataset[1] for dataset in datasets if len(dataset[1])>=1][0])
    print(dataset_name)

    processes = []
    for dataset in dataset_name:
        print("Dataset: ", dataset)
        proc = Process(target=agent, args=(path, dataset,red, seg, folder))
        processes.append(proc)
        proc.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    cli()
    #cli('./test/', PAA=True, n_seg=0.75)

