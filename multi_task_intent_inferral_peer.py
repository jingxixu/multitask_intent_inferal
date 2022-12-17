# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import torch

"""# utils"""

import numpy as np


def clean_dataframe(df):
    X_df = keep_columns(df, [f'emg'])
    X = X_df.to_numpy()
    y_df = keep_columns(df, ['gt'])
    y = y_df.to_numpy().squeeze()
    return X, y


def drop_columns(df, tuple_of_columns):
    """
    Given a dataframe, and a tuple of column names, this function will search
    through the dataframe and drop the columns which contain a string from the
    list of the undesired columns. All other columns are kept
    """
    if len(tuple_of_columns) >= 1:
        cols = df.columns[df.columns.to_series().str.contains('|'.join(tuple_of_columns))]
        return df.drop(columns=cols)
    return df


def keep_columns(df, tuple_of_columns):
    """
    Given a dataframe, and a tuple of column names, this function will search
    through the dataframe and keep only columns which contain a string from the
    list of the desired columns. All other columns are removed
    """
    if len(tuple_of_columns) >= 1:
        cols = df.columns[df.columns.to_series().str.contains('|'.join(tuple_of_columns))]
        return df[cols]
    return df


def random_split(X, y, split=0.8):
    assert len(X) == len(y)
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    num_train_samples = round(num_samples * split)
    X_train, X_test = X[indices[:num_train_samples]], X[indices[num_train_samples:]]
    y_train, y_test = y[indices[:num_train_samples]], y[indices[num_train_samples:]]
    return X_train, y_train, X_test, y_test

def test(w, k, X, Y, alg, sparse=False):
    # return 0
    err_count = np.zeros((k, 1))
    for i in range(X.shape[0]):
        if sparse:
            x = X.getrow(i).toarray()
            tid = int(x[0][0])
            x = x[0][1:]
        else:
            x = X[i][1:]
            tid = int(X[i][0])
        x = np.concatenate((x, np.asarray([1])))
        x = x / np.linalg.norm(x)
#         if alg == "pooled":
#             y_hat = np.sign(w[0].dot(x))
#         else:
        y_hat = np.sign(w[tid].dot(x))
        if y_hat == 0:
            y_hat = 1
        yt = Y[i]
        if yt == 0:
            yt = -1
        if yt != y_hat:
            err_count[tid] += 1
    return 1 - np.sum(err_count) / X.shape[0]

def print_summary(model, acc_list, query_count, err_count, total_count, run):
    """
    :param model: name of this model
    :param acc_list: list of run elements
    :param query_count: (k, run) ndarray
    :param err_count: (k, run) ndarray
    :param run: number of runs
    """
    accuracy = np.average(acc_list)
    query = np.sum(query_count) / run
    mistake_rates = np.sum(err_count, axis=0)/np.sum(total_count, axis=0)
    mistake_rate = np.average(mistake_rates)
    acc_std_err = get_std_error(acc_list)
    query_std_err = get_std_error(np.sum(query_count, axis=0))
    mistake_std_err = get_std_error(mistake_rates)

    print()
    print(model)
    print("accuracy: {:.4f}, acc_std_err: {:.4f}, query: {:.1f}, query_std_err: {:.1f}, mistakes: {:.4f}, mistakes_std_err: {:.4f}"\
          .format(accuracy, acc_std_err, query, query_std_err, mistake_rate, mistake_std_err))
    return accuracy, query, mistake_rate

def get_std_error(a):
    """
    Get standard error, given samples.
    :param a: 1d array like
    """
    a = np.array(a)
    n = len(a)
    # return np.std(a, ddof=1)/np.sqrt(n)
    # Note now actually just get std deviation
    return np.std(a, ddof=1)

# then remove begin 10 and end 10
def moving_avg(y_pred_prob, kernel_size = 20):
  kernel = np.ones(kernel_size) / kernel_size
  y_pred_prob_avg = np.zeros_like(y_pred_prob)
  y_pred_prob_avg[:,0] = np.convolve(y_pred_prob[:,0], kernel, mode='same')
  y_pred_prob_avg[:,1] = np.convolve(y_pred_prob[:,1], kernel, mode='same')
  y_pred_prob_avg[:,2] = np.convolve(y_pred_prob[:,2], kernel, mode='same')
  return y_pred_prob_avg[10:-10]

"""# data """

# load data from file
import os

data_file_dir= 'collected_data/2022_12_14/'

def xy_from_file(filename):
    df = pd.read_csv(os.path.join(data_file_dir,filename), index_col=0)
    return clean_dataframe(df)

X_jx00, y_jx00 = xy_from_file('jx_00.csv')
X_jx01, y_jx01 = xy_from_file('jx_01.csv') # test 1
X_jx30, y_jx30 = xy_from_file('jx_30.csv')
X_jx60, y_jx60 = xy_from_file('jx_60.csv')
X_ss00, y_ss00 = xy_from_file('ss_00.csv')
X_ss01, y_ss01 = xy_from_file('ss_01.csv') # test 2
X_ss30, y_ss30 = xy_from_file('ss_30.csv')
X_ss60, y_ss60 = xy_from_file('ss_60.csv')
X_ss02, y_ss02 = xy_from_file('ss_02_short.csv')
X_jx02, y_jx02 = xy_from_file('jx_02_short.csv')

trainX_ftjx = X_jx02
testX_ftjx = X_jx01
trainY_ftjx = y_jx02
testY_ftjx = y_jx01
trainX_ftss = X_ss02
testX_ftss = X_ss01
trainY_ftss = y_ss02
testY_ftss = y_ss01



trainset_ftjx = list(zip(trainX_ftjx, trainY_ftjx))
testset_ftjx = list(zip(testX_ftjx, testY_ftjx))

trainset_ftss = list(zip(trainX_ftss, trainY_ftss))
testset_ftss = list(zip(testX_ftss, testY_ftss))

# get the data loader for full previous data
X = np.vstack((X_jx00, X_jx30, X_jx60, X_ss00, X_ss30, X_ss60, X_jx02, X_ss02))
Y = np.hstack((y_jx00, y_jx30, y_jx60, y_ss00, y_ss30, y_ss60, y_jx02, y_ss02))
fullset = list(zip(X, Y))

train_num = int(len(fullset)*4/5)
full_train, full_test = torch.utils.data.random_split(fullset, [train_num, len(fullset) - train_num], generator=torch.Generator().manual_seed(42))

# get the data loaders for all three datasets
def train_test_from_xy(X,y):
  set_ = list(zip(X, y))
  train_num = int(len(set_)*4/5)
  return torch.utils.data.random_split(set_, [train_num, len(set_) - train_num], generator=torch.Generator().manual_seed(42))

train_jx00, test_jx00 = train_test_from_xy(X_jx00, y_jx00)
train_jx01, test_jx01 = train_test_from_xy(X_jx01, y_jx01)
train_jx30, test_jx30 = train_test_from_xy(X_jx30, y_jx30)
train_jx60, test_jx60 = train_test_from_xy(X_jx60, y_jx60)
train_ss00, test_ss00 = train_test_from_xy(X_ss00, y_ss00)
train_ss01, test_ss01 = train_test_from_xy(X_ss01, y_ss01)
train_ss30, test_ss30 = train_test_from_xy(X_ss30, y_ss30)
train_ss60, test_ss60 = train_test_from_xy(X_ss60, y_ss60)

from sklearn.preprocessing import StandardScaler
def get_train_test_norm(trainset, testset):
  train_X = [i for i,j in trainset]
  train_y = [j for i,j in trainset]
  scaler = StandardScaler()
  scaler.fit(train_X)
  train_X_norm = scaler.transform(train_X)
  test_X = [i for i,j in testset]
  test_y = [j for i,j in testset]
  test_X_norm = scaler.transform(test_X)

  return train_X_norm, train_y, test_X_norm, test_y, scaler


def get_train_test_norm_scaled(trainset, testset, scaler):
  train_X = [i for i,j in trainset]
  train_y = [j for i,j in trainset]
  train_X_norm = scaler.transform(train_X)
  test_X = [i for i,j in testset]
  test_y = [j for i,j in testset]
  test_X_norm = scaler.transform(test_X)

  return train_X_norm, train_y, test_X_norm, test_y

# jxft with jx00
X = np.vstack((X_jx00, X_jx02))
Y = np.hstack((y_jx00, y_jx02))
aug_jx01 = list(zip(X, Y))

train_num = int(len(aug_jx01)*4/5)
train_jx01a, test_jx01a = torch.utils.data.random_split(aug_jx01, [train_num, len(aug_jx01) - train_num], generator=torch.Generator().manual_seed(42))

"""# Sklearn"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def LR_scaler_with_train_test(trainset, testset):
  train_X_norm, train_y, test_X_norm, test_y, scaler = get_train_test_norm(trainset, testset)
  clf = LogisticRegression(random_state=0).fit(train_X_norm, train_y)
  # print(clf.score(test_X_norm, test_y))
  return clf, scaler

clf_jx00, scaler_jx00 = LR_scaler_with_train_test(train_jx00, test_jx00)
clf_jx01, scaler_jx01 = LR_scaler_with_train_test(train_jx01, test_jx01)
clf_jx30, scaler_jx30 = LR_scaler_with_train_test(train_jx30, test_jx30)
clf_jx60, scaler_jx60 = LR_scaler_with_train_test(train_jx60, test_jx60)
clf_ss00, scaler_ss00 = LR_scaler_with_train_test(train_ss00, test_ss00)
clf_ss01, scaler_ss01 = LR_scaler_with_train_test(train_ss01, test_ss01)
clf_ss30, scaler_ss30 = LR_scaler_with_train_test(train_ss30, test_ss30)
clf_ss60, scaler_ss60 = LR_scaler_with_train_test(train_ss60, test_ss60)

# self acc
def get_self_acc():
  train_X_norm, train_y, test_X_norm, test_y, scaler_ftjx = get_train_test_norm(trainset_ftjx, testset_ftjx)
  clf_selfjx = LogisticRegression(random_state=0).fit(train_X_norm, train_y)
  train_X_norm, train_y, test_X_norm, test_y = get_train_test_norm_scaled(trainset_ftjx, testset_ftjx, scaler_ftjx)
  y_pred_list_self = clf_selfjx.predict(test_X_norm)
  y_pred_prob_self = clf_selfjx.predict_proba(test_X_norm)
  y_pred_prob_self_avg = moving_avg(y_pred_prob_self)
  y_pred_list_self_avg = np.argmax(y_pred_prob_self_avg, axis=1)
  return scaler_ftjx, clf_selfjx, accuracy_score(y_pred_list_self_avg, testY_ftjx[10:-10])
  
scaler_ftjx, clf_selfjx, self_acc = get_self_acc()

# full acc
def get_full_acc():
  train_X_norm, train_y, test_X_norm, test_y, scaler_full = get_train_test_norm(full_train, full_test)
  clf_full = LogisticRegression(random_state=0).fit(train_X_norm, train_y)
  train_X_norm, train_y, test_X_norm, test_y = get_train_test_norm_scaled(trainset_ftjx, testset_ftjx, scaler_full)
  y_pred_prob_full = clf_full.predict_proba(test_X_norm)
  y_pred_prob_full_avg = moving_avg(y_pred_prob_full)
  y_pred_list_full_avg = np.argmax(y_pred_prob_full_avg, axis=1)
  return scaler_full, clf_full, accuracy_score(y_pred_list_full_avg, testY_ftjx[10:-10])
scaler_full, clf_full, full_acc = get_full_acc()

"""# Learn Similiarity"""

import torch.nn as nn
criterion = nn.CrossEntropyLoss()
class Peer_Similarity_LR:
    def __init__(self, b_1, b_2, lambda_, T, k, d, model_list, scaler_list):
        # b_2 >= b_1 > 0, lambda > 0
        self.b_1 = b_1
        self.b_2 = b_2
        self.lambda_ = lambda_
        self.T = T
        self.d = d
        self.k = k
        self.tau = np.ones((k, k))/(k-1) - np.eye(k) * ((k-2) / (k-1))
        self.model_list = model_list
        self.scaler_list = scaler_list
        self.similarity_list = []

    def fit(self, X, Y, X_test, Y_test):
        k = self.k
        fea = self.d

        acc = []
        for t in range(self.T):
            # init 
            shuffle = np.random.permutation(X.shape[0])
            self.tau = np.ones((k, k))/(k-1) - np.eye(k)  / (k-1)
            w = np.zeros((k, fea))
            
            # loop T times
            for i in range(X.shape[0]):

                x_origin = X[shuffle[i]]
                tid = self.k-1

                # get f_t
                y_hat_peer = []
                y_hat_peer_tensor = []
                for idx, model in enumerate(self.model_list):
                  x = self.scaler_list[idx].transform(x_origin.reshape(1,8))
                  x_tensor = torch.from_numpy(x.reshape(1,8)).float()
                  y_hat = model.predict_proba(x_tensor)
                  y_hat_peer.append(y_hat)
                  y_hat_peer_tensor.append(torch.from_numpy(y_hat).float())

                yt = Y[shuffle[i]]

                
                other_tasks = [i for i in range(k) if i != tid]

                self.similarity_list.append(self.tau[tid][other_tasks]) 

                yt_tensor = torch.from_numpy(np.array([yt])).long()
                # loss, cross loss
                l_kmt_de_list = []
                for y_hat in y_hat_peer_tensor:
                  l_kmt_de_list.append(criterion(y_hat, yt_tensor).numpy())
                l_kmt_denominator = np.array(l_kmt_de_list)

                denominator = np.multiply(self.tau[tid][other_tasks], np.exp(-1 * l_kmt_denominator / self.lambda_))
                self.tau[tid][other_tasks] = denominator / np.sum(denominator)  

                             

            self.similarity_list.append(self.tau[tid][other_tasks]) 
        return self.tau, self.similarity_list

    def inference_with_similarity(self, model_ft, scaler_ft, X, y_test):
        peer_count = 0
        self_count = 0
        correct_count = 0
        y_pred_list = []
        peer_prob_list = []
        for i in range(X.shape[0]):
            x_origin = X[i]
            tid = self.k-1
        
            y_pred = 0
            # get f_t
            y_hat_peer = []
            y_hat_peer_tensor = []
            for idx, model in enumerate(self.model_list):
              x = self.scaler_list[idx].transform(x_origin.reshape(1,8))
              x_tensor = torch.from_numpy(x.reshape(1,8)).float()
              y_hat = model.predict_proba(x_tensor)
              y_hat_peer.append(y_hat)
              y_hat_peer_tensor.append(torch.from_numpy(y_hat).float())

            x = scaler_ft.transform(x_origin.reshape(1,8))
            x_tensor = torch.from_numpy(x.reshape(1,8)).float()
            y_hat = model_ft.predict_proba(x_tensor)
            y_peer = (np.array(y_hat_peer).reshape(6,3).T*self.tau[self.k-1,:-1]).T
            y_peer = np.sum(y_peer, axis=0)

            peer_prob_list.append(y_peer)

            flag = np.random.uniform(0,1)            
            if flag > (self.b_1 / (self.b_1+np.max(y_hat[0]))):
              y_pred = np.argmax(y_hat[0])
              peer_count += 1
            else:
              y_pred = np.argmax(y_peer)
              self_count += 1

            y_pred_list.append(y_pred)
            if y_pred == y_test[i]:
              correct_count += 1

        return peer_count, self_count, float(correct_count)/X.shape[0], np.array(y_pred_list), np.array(peer_prob_list)

Similarity_Learner_LR = Peer_Similarity_LR(3, 10, 1, 1, 7, 9, 
                                           [clf_jx30, clf_jx00, clf_jx60, clf_ss00, clf_ss30, clf_ss60],
                                           [scaler_jx30, scaler_jx00, scaler_jx60, scaler_ss00, scaler_ss30, scaler_ss60])

tau, similarity_list = Similarity_Learner_LR.fit(trainX_ftjx, trainY_ftjx, testX_ftjx, testY_ftjx)

peer_count, self_count, ours_acc, y_pred_list_peer, y_pred_prob_peer = Similarity_Learner_LR.inference_with_similarity(clf_selfjx, scaler_ftjx, testX_ftjx, testY_ftjx)
# probablity moving average
y_pred_prob_peer_avg = moving_avg(y_pred_prob_peer)
y_pred_list_peer_avg = np.argmax(y_pred_prob_peer_avg, axis=1)
ours_acc = accuracy_score(y_pred_list_peer_avg, testY_ftjx[10:-10])

print("Self accuracy:", self_acc)
print("Fine-tune accuracy:", full_acc)
print("Self+Peers(Ours) accuracy:", ours_acc)
