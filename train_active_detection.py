# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
from time import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

import torch

from models.utils import parse_args_anomaly
from sklearn.metrics import roc_auc_score
from models.Active_ADetection import Anomaly_Detection
from models.DataSet import ODDSDataset

if __name__=='__main__':
    # Define data file and read X and y
    data_root = './data'
    save_dir = './results'


    data_names = ['arrhythmia.mat',
                    'cardio.mat',
                    'glass.mat',
                    'ionosphere.mat',
                    'letter.mat',
                    'lympho.mat',
                    'mnist.mat',
                    'musk.mat',
                    'optdigits.mat',
                    'pendigits.mat',
                    'pima.mat',
                    'satellite.mat',
                    'satimage-2.mat',
                    'shuttle.mat',
                    'vowels.mat',
                    'annthyroid.mat',
                    'campaign.mat',
                    'celeba.mat',
                    'fraud.mat',
                    'donors.mat'
                    ]
    # define the number of iterations
    n_ite = 10
    n_classifiers = 1

    df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
                'EAL-GAN']

    # initialize the container for saving the results
    roc_df = pd.DataFrame(columns=df_columns)
    gmean_df = pd.DataFrame(columns=df_columns)
    args = parse_args_anomaly()

    for data_name in data_names:
        mat = loadmat(os.path.join(data_root, data_name))

        X = mat['X']
        y = mat['y'].ravel()
        y = y.astype(np.long)

        outliers_fraction = np.count_nonzero(y) / len(y)
        print(outliers_fraction)
        outliers_percentage = round(outliers_fraction * 100, ndigits=4)

        # construct containers for saving results
        roc_list = [data_name[:-4], X.shape[0], X.shape[1], outliers_percentage]
        gmean_list = [data_name[:-4], X.shape[0], X.shape[1], outliers_percentage]
        
        roc_mat = np.zeros([n_ite, n_classifiers])
        gmean_mat = np.zeros([n_ite, n_classifiers])
        
        for i in range(n_ite):
            print("\n... Processing", data_name[:-4], '...', 'Iteration', i + 1)
            random_state = np.random.RandomState(i)
            train_data = ODDSDataset(root=data_root, dataset_name=data_name, train=True, test_size=0.3, random_state=random_state)
            test_data = ODDSDataset(root=data_root, dataset_name=data_name, train=False, test_size=0.3, random_state=random_state)
            
            t0 = time()
            #todo: add my method
            mlp = Anomaly_Detection(args, train_data, test_data)
            best_auc, best_gmean = mlp.fit()
            #clf.fit(X_train_norm)
            #test_scores = clf.decision_function(X_test_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)
            
            
            print('AUC:%.4f, Gmean:%.4f ' % (best_auc, best_gmean))

            roc_mat[i, 0] = best_auc
            gmean_mat[i, 0] = best_gmean

        roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
        temp_df = pd.DataFrame(roc_list).transpose()
        temp_df.columns = df_columns
        roc_df = pd.concat([roc_df, temp_df], axis=0)

        gmean_list = gmean_list + np.mean(gmean_mat, axis=0).tolist()
        temp_df = pd.DataFrame(gmean_list).transpose()
        temp_df.columns = df_columns
        gmean_df = pd.concat([gmean_df, temp_df], axis=0)

        # Save the results for each run
        save_path1 = os.path.join(save_dir, "AUC_Active_Anomaly.csv")
        save_path2 = os.path.join(save_dir, "Gmean_Active_Anomaly.csv")

        roc_df.to_csv(save_path1, index=False, float_format='%.4f')
        gmean_df.to_csv(save_path2, index=False, float_format='%.4f')