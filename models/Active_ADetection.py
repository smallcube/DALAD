from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop, SGD

import numpy as np

from models.utils import AUC_and_Gmean


class Anomaly_Detection(nn.Module):
    def __init__(self, args, train_data, test_data):
        super(Anomaly_Detection, self).__init__()
        self.args = args
        self.train_data = train_data
        self.test_data = test_data
        self.train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        self.data_size, self.feat_dim = train_data.get_dims()
        self.hidden_list = [128, 64, 8]
        self.activation = nn.Sigmoid()

        self.encoder = nn.Sequential(nn.Linear(self.feat_dim, 256), self.activation,
                                    nn.Linear(256, 64), self.activation,
                                    nn.Linear(64, 8), self.activation)
        
        self.decoder = nn.Sequential(nn.Linear(8, 64), self.activation,
                                    nn.Linear(64, 256), self.activation,
                                    nn.Linear(256, self.feat_dim), self.activation)
        
        self.detector = nn.Linear(8+1, 1, bias=True)

        #self.optimizer_unsupervised =  RMSprop(list(self.encoder.parameters())+list(self.decoder.parameters()) + list(self.detector.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0005)
        #self.optimizer_supervised =  RMSprop(list(self.encoder.parameters())+list(self.decoder.parameters()) + list(self.detector.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0005)
        
        self.optimizer_unsupervised =  SGD(list(self.encoder.parameters())+list(self.decoder.parameters()) + list(self.detector.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0005)
        self.optimizer_supervised =  SGD(list(self.encoder.parameters())+list(self.decoder.parameters()) + list(self.detector.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    

    
    def fit(self):
        best_record = 0
        best_auc = 0
        best_gmean = 0
        for epoch in range(self.args.max_epochs_stage_1):
            train_auc, train_gmean, test_auc, test_gmean = self.train_one_epoch_unsupervised()
            if train_auc*train_gmean>best_record:
                best_record = train_auc*train_gmean
                best_auc = test_auc
                best_gmean = test_gmean
            
            print("Stage 1  Epoch %3d: Train_AUC=%.4f   Train_Gmean=%.4f   Test_AUC=%.4f   Test_Gmean=%.4f" % (epoch, train_auc, train_gmean, test_auc, test_gmean))
        
        #for p in self.encoder.parameters():
        #    p.requires_grad = False
        #for p in self.decoder.parameters():
        #    p.requires_grad = False

        best_record = 0
        for epoch in range(self.args.max_epochs_stage_2):
            train_auc, train_gmean, test_auc, test_gmean = self.train_one_epoch_supervised()
            if train_auc*train_gmean>=best_record:
                best_record = train_auc*train_gmean
                best_auc = test_auc
                best_gmean = test_gmean

            print("Stage 2  Epoch %3d: Train_AUC=%.4f   Train_Gmean=%.4f   Test_AUC=%.4f   Test_Gmean=%.4f" % (epoch, train_auc, train_gmean, test_auc, test_gmean))
        
        return best_auc, best_gmean
    
    def train_one_epoch_unsupervised(self):
        for (x, y, index) in self.train_loader:
            noise = torch.empty_like(x).normal_(0, 1)
            representation = self.encoder(x+noise)

            x_reconstructed = self.decoder(representation)
            loss = (x_reconstructed-x)**2
            loss = loss.sum(dim=1).mean()
            
            self.optimizer_unsupervised.zero_grad()
            loss.backward()
            self.optimizer_unsupervised.step()
        
        y_train, y_scores_train = self.predict_unsupervised(self.train_loader)
        auc_train, gmean_train = AUC_and_Gmean(y_train, y_scores_train)

        y_test, y_scores_test = self.predict_unsupervised(self.test_loader)
        auc_test, gmean_test = AUC_and_Gmean(y_test, y_scores_test)

        return auc_train, gmean_train, auc_test, gmean_test
        
    
    def predict_unsupervised(self, dataloader):
        p = []
        y = []
        
        for data in dataloader:
            x = data[0]
            targets = data[1]

            r = self.encoder(x)
            pt = self.decoder(r)
            pt = (pt-x)**2
            pt = pt.sum(dim=1)
            p += [pt]
            y += [targets]
        
        p = torch.cat(p, 0).cpu().detach().numpy()
        y = torch.cat(y, 0).cpu().detach().numpy()
        return y, p
    

    def fit_supervised(self):
        best_record = 0
        best_auc = 0
        best_gmean = 0
        for epoch in range(self.args.max_epochs):
            train_auc, train_gmean, test_auc, test_gmean = self.train_one_epoch_supervised()
            if train_auc*train_gmean>=best_record:
                best_record = train_auc*train_gmean
                best_auc = test_auc
                best_gmean = test_gmean
        return best_auc, best_gmean
    
    def train_one_epoch_supervised(self):
        for (x, y, index) in self.train_loader:
            representation = self.encoder(x)

            x_reconstructed = self.decoder(representation)
            x_representation = (x_reconstructed-x)**2
            x_representation = x_representation.sum(dim=1).view(-1, 1)

            x_representation = torch.cat([x_representation, representation], 1)

            loss_unsupervised = (x_reconstructed-x)**2
            loss_unsupervised = loss_unsupervised.sum(dim=1).mean()

            #print("x_representation,     ", x_representation.shape)

            score = self.detector(x_representation)
            score = score.sigmoid()

            #Active Learning
            self.args.top_K = x.shape[0]
            _, idx = torch.sort(score, descending=True)
            y_predicted = score[idx[0:self.args.top_K]].view(-1, 1)
            y_true = y[idx[0:self.args.top_K]]

            #print("score.shape", score.shape, "    ", y_predicted.dtype, "     ", y_true.dtype)
            #print(y_predicted.mean())
            #print(y_true)
            loss = F.binary_cross_entropy(y_predicted, y_true.float()) + loss_unsupervised
            #print("loss=", loss.item())
            
            self.optimizer_supervised.zero_grad()
            loss.backward()
            self.optimizer_supervised.step()
            
        y_train, y_scores_train = self.predict_supervised(self.train_loader)
        auc_train, gmean_train = AUC_and_Gmean(y_train, y_scores_train)

        y_test, y_scores_test = self.predict_supervised(self.test_loader)
        auc_test, gmean_test = AUC_and_Gmean(y_test, y_scores_test)

        return auc_train, gmean_train, auc_test, gmean_test
        
    
    def predict_supervised(self, dataloader):
        p = []
        y = []
        
        for data in dataloader:
            x = data[0]
            targets = data[1]

            r = self.encoder(x)
            pt = self.decoder(r)
            pt = (pt-x)**2
            pt = pt.sum(dim=1).view(-1, 1)
            pt = torch.cat([pt, r], 1)
            pt = self.detector(pt)
            pt = pt.sigmoid()

            p += [pt]
            y += [targets]
        
        p = torch.cat(p, 0).cpu().detach().numpy()
        y = torch.cat(y, 0).cpu().detach().numpy()
        return y, p
