import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from numpy import *
import argparse

from numpy import percentile
import numbers
import math

import sklearn
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length

from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics   import roc_auc_score

def get_gmean(y, y_pred, threshold=0.5):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    
    Returns
    -------
    Gmean: float
    """
    #y_pred = get_label_n(y, y_pred)
    y = y.reshape(-1, )
    y_pred = y_pred.reshape(-1, )
    y_pred = (y_pred >= threshold).astype('int')
    ones_all = (y==1).sum()
    ones_correct = ((y==1) & (y_pred==1)).sum()

    zeros_all = (y==0).sum()
    zeros_correct = ((y==0) * (y_pred==0)).sum()
    
    #print("one_correst=", ones_correct, "   zeros=", zeros_correct, " all=", y.shape[0])
    Gmean = np.sqrt((1.0*ones_correct/ones_all) * (1.0*zeros_correct/zeros_all))
    #Gmean *= np.sqrt

    return Gmean

def AUC_and_Gmean(y_test, y_scores):
    #print(y_test)
    #print(y_scores)
    auc = round(roc_auc_score(y_test, y_scores), ndigits=4)
    gmean = round(get_gmean(y_test, y_scores, 0.5), ndigits=4)

    return auc, gmean


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ensemble.")
    parser.add_argument('--data_root', default='./data/', type=str,
                        help='the data folder path.')
    parser.add_argument('--data_name', default='Annthyroid', type=str,
                        help='Input data name.')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--activate_ratio', type=float, default=None,
                        help='Learning rate.')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size.')
    
    parser.add_argument('--hidden_depth', type=int, default=1,
                        help='hidden_layer number in net.')
    return parser.parse_args()

def parse_args_anomaly():
    parser = argparse.ArgumentParser(description="Active Anomaly Detection.")
    parser.add_argument('--data_root', default='./data/', type=str,
                        help='the data folder path.')
    parser.add_argument('--data_name', default='Annthyroid', type=str,
                        help='Input data name.')
    parser.add_argument('--max_epochs_stage_1', type=int, default=50,
                        help='epochs for stage 1')
    parser.add_argument('--max_epochs_stage_2', type=int, default=20,
                        help='epochs for stage 2')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--top_K', type=float, default=30,
                        help='top K instances for active learning')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size.')
    
    
    return parser.parse_args()



