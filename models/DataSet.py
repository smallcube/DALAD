from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import pandas as pd

import os
import torch
import numpy as np
from imblearn.over_sampling import SMOTE
import smote_variants as sv

class ODDSDataset(Dataset):
    def __init__(self, root: str, dataset_name: str, train=True, test_size=0.4, active_ratio=None, random_state=None, oversample=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        self.file_name = self.dataset_name
        self.data_file = self.root / self.file_name

        mat = loadmat(self.data_file)
        X = mat['X']
        y = mat['y'].ravel()

        '''
        idx_norm = y == 0
        idx_out = y == 1
        # 60% data for training and 40% for testing; keep outlier ratio
        
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],
                                                                                test_size=0.4,
                                                                                random_state=random_state)
        X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],
                                                                            test_size=0.4,
                                                                            random_state=random_state)
        X_train = np.concatenate((X_train_norm, X_train_out))
        X_test = np.concatenate((X_test_norm, X_test_out))
        y_train = np.concatenate((y_train_norm, y_train_out))
        y_test = np.concatenate((y_test_norm, y_test_out))
        '''
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Scale to range [0,1]
        #minmax_scaler = MinMaxScaler().fit(X_train_stand)
        #X_train_scaled = minmax_scaler.transform(X_train_stand)
        #X_test_scaled = minmax_scaler.transform(X_test_stand)

        X_train_pandas = pd.DataFrame(X_train_scaled)
        X_test_pandas = pd.DataFrame(X_test_scaled)
        X_train_pandas.fillna(X_train_pandas.mean(), inplace=True)
        X_test_pandas.fillna(X_train_pandas.mean(), inplace=True)
        X_train_scaled = X_train_pandas.values
        X_test_scaled = X_test_pandas.values

        if self.train:
            if active_ratio is not None:
                X_train_scaled, _, y_train, _ = \
                            train_test_split(X_train_scaled, y_train, test_size=active_ratio, random_state=random_state, stratify=y_train)

            if oversample:
                #smote = SMOTE()
                oversampler= sv.distance_SMOTE()

                # X_samp and y_samp contain the oversampled dataset
                X_train_scaled, y_train = oversampler.sample(X_train_scaled, y_train)
                #X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                #por = (y_train==1).sum()/y_train.shape[0]
                #print("por=", por)
            #print(y_train[0:10])

            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target = self.data[index], int(self.targets[index])

        return sample, target, index

    def __len__(self):
        return len(self.data)
        
    def get_dims(self):
        return self.data.shape[0], self.data.shape[1]

    def get_ratio(self):
        ratio = torch.sum(self.targets==1)/self.targets.shape[0]
        return ratio

    def get_labels(self):
        return self.targets
