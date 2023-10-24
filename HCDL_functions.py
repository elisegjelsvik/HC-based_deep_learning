# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:29:52 2023

@author: Elise Lunde Gjelsvik
elise.lunde.gjelsvik@nmbu.no
"""
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from fcmeans import FCM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.io
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA 
from sklearn.naive_bayes import GaussianNB
import pickle
import tensorflow as tf
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from scipy.special import eval_legendre
from scipy import signal
import numpy.linalg as la

# =============================================================================
# Explained variance PLS
# =============================================================================
def explained_variance_pls(model, X):
    Xs = X# - np.mean(X, axis=0)
    T = model.transform(X) # scores
    P = model.x_loadings_  # loadings
    total_msse = np.mean(np.sum(np.square(Xs), axis=0)) # Mean over all variables
    n = T.shape[1]
    exp_var = np.zeros(n)
    for i in range(n):
        X_hat = T[:, i:i+1] @ P[:, i:i+1].T
        msse = np.mean(np.sum(np.square(Xs - X_hat), axis=0))
        exp_var[i] = (total_msse - msse)/total_msse
    return exp_var

# =============================================================================
# EMSC preprosessering
# =============================================================================
def emsc(X, ref, d=0):
    """Implementation using Legendre polynomials from scipy"""
    n = len(ref)                 # Number of features
    x = np.linspace(-1, 1, n)
    P = np.zeros([n, d+2])       # Polynomial basis vectors 
    P[:, 0] = ref                # First column is the reference spectrum

    # Add the legendre polynomials
    for i in range(0, d+1):
            P[:, i+1] = eval_legendre(i, x)
    # Calculate the coefficients
    coeffs, _, _, _ = la.lstsq(P, X.T, rcond=None)
    # Get the transformed spectra
    Xprep = X.copy()
    Xprep -= (P[:, 1:] @ coeffs[1:, :]).T
    Xprep /= coeffs[0, :].reshape(-1, 1)
    return Xprep

# =============================================================================
# Removes clusters with less than 10 samples
# =============================================================================
def remove_small_clusters(fcm_model, df, no_clusters):
    fcm_cluster_prob_df = pd.DataFrame(fcm_model.u.argsort())
    fcm_labels = np.array(fcm_cluster_prob_df.loc[:, no_clusters-1])
    
    label_dict = Counter(fcm_labels)
    label_dict = dict(label_dict)
    
    for i in range(0, no_clusters):
        if i in label_dict.keys():
            pass
        else:
            label_dict[i] = 0
    
    new_fcm_labels = []
    for s in range(len(df)):
        sample = df.iloc[s]
        sample_label = fcm_labels[s]
        new_cluster = fcm_cluster_prob_df.iloc[s]
        
        if label_dict[sample_label] < 10:
            new_label = new_cluster.iloc[-2] 
            v = 2
            
            while label_dict[new_label] < 10:
                new_label = new_cluster.iloc[-v]
                v += 1
        
            else:
                new_fcm_labels.append(new_label)
        
        else:
            new_fcm_labels.append(sample_label)
    
    return np.array(new_fcm_labels), label_dict
    

def remove_small_clusters_test(X_test, X_test_scores, model, training_cluster_dict, new_training_labels, no_clusters):
    df_test_labels = model.soft_predict(X_test_scores)
    df_test_labels = pd.DataFrame(df_test_labels.argsort())
    test_label = np.array(df_test_labels.iloc[:, no_clusters-1])
    
    key_list = list(np.unique(new_training_labels))
    
    new_test_labels = []
    for s in range(len(X_test)):
        sample = X_test.iloc[s]
        sample_label = test_label[s]
        new_cluster = df_test_labels.iloc[s]
        
        if training_cluster_dict[sample_label] < 10:
            new_label = new_cluster.iloc[-2]

            v = 2
            while new_label not in key_list:
                new_label = new_cluster.iloc[-v]
                v += 1
            else:
                new_test_labels.append(new_label)
        else:
            new_test_labels.append(sample_label)
            
    return np.array(new_test_labels)

# =============================================================================
# Local predictions PLSR
# =============================================================================
def plsr_local_predictions(std_list, labels, data, y, no_clusters, local_pls):
    test_index = np.unique(labels)
    local_predictions = []
    local_index = []
    for p in range(0, no_clusters):
        test_class = data[labels == p]
        local_index.append(test_class.index)
        pls = local_pls[p]
        test_y = y[labels == p]
        
        if len(test_class) > 1:
            test_class = std_list[p].transform(test_class)
            local_predictions.append(pls.predict(test_class))
            
        elif len(test_class) == 1:
            test_class = std_list[p].transform(test_class)
            print('Only one sample in cluster %.0f' % p)
            local_predictions.append(pls.predict(test_class))
            
        else:
            print('No samples in cluster %.0f' % p)
    
    local_predictions_df = pd.DataFrame(np.concatenate(local_predictions))
    local_index = np.concatenate(local_index)
    local_predictions_df.index = local_index
    return local_predictions_df


# =============================================================================
# Local predictions CNN
# =============================================================================
def cnn_local_prediction(std_list, X_test, test_labels, y, no_clusters, local_cnn):
    test_index = np.unique(test_labels)
    local_predictions = []
    local_index = []
    for p in range(0, no_clusters):
        test_cluster = X_test[test_labels == p]
        local_index.append(test_cluster.index)
        
        cnn = local_cnn[p]
        test_y = y[test_labels == p]
        
        if len(test_cluster) > 1:
            test_cluster = std_list[p].transform(test_cluster)
            
            n_features = 1
            X_test_reshape = np.array(test_cluster)
            X_test_reshape = X_test_reshape.reshape(X_test_reshape.shape[0], X_test_reshape.shape[1], n_features)
            
            y_pred = cnn.predict(X_test_reshape)
            local_predictions.append(y_pred)
            # print(r2_score(test_y, y_pred))
        elif len(test_cluster) == 1:
            print('Only one sample in cluster %.0f' % p)
            test_cluster = std_list[p].transform(test_cluster)
            
            n_features = 1
            X_test_reshape = np.array(test_cluster)
            X_test_reshape = X_test_reshape.reshape(X_test_reshape.shape[0], X_test_reshape.shape[1], n_features)
            
            y_pred = cnn.predict(X_test_reshape)
            local_predictions.append(y_pred)
        else:
            print('No samples in cluster %.0f' % p)
        
    local_predictions_df = pd.DataFrame(np.concatenate(local_predictions))
    local_index = np.concatenate(local_index)
    local_predictions_df.index = local_index
    return local_predictions_df

# =============================================================================
# Local predictions RNN
# =============================================================================
def rnn_local_prediction(std_list, X_test, test_labels, y, no_clusters, local_rnn):
    test_index = np.unique(test_labels)
    local_predictions = []
    local_index = []
    for p in range(0, no_clusters):
        test_cluster = X_test[test_labels == p]
        local_index.append(test_cluster.index)
        
        test_y = y[test_labels == p]
        rnn = local_rnn[p]
        
        if len(test_cluster) > 1:
            test_cluster = std_list[p].transform(test_cluster)
        
            n_features = 1
            X_test_reshape = np.array(test_cluster)
            X_test_reshape = X_test_reshape.reshape(X_test_reshape.shape[0], n_features, X_test_reshape.shape[1])        
        
            y_pred = rnn.predict(X_test_reshape)
            local_predictions.append(y_pred)
        elif len(test_cluster) == 1:
            print('Only one sample in cluster %.0f' % p)
            
            test_cluster = std_list[p].transform(test_cluster)
            
            n_features = 1
            X_test_reshape = np.array(test_cluster)
            X_test_reshape = X_test_reshape.reshape(X_test_reshape.shape[0], n_features, X_test_reshape.shape[1])        
            
            y_pred = rnn.predict(X_test_reshape)
            local_predictions.append(y_pred)        
            # print(r2_score(test_y, y_pred))
        else:
            print('No samples in cluster %.0f' % p)
        
    local_predictions_df = pd.DataFrame(np.concatenate(local_predictions))
    local_index = np.concatenate(local_index)
    local_predictions_df.index = local_index
    return local_predictions_df

# =============================================================================
# Local predictions SVR
# =============================================================================
def svm_local_predictions(std_list, test_labels, X_test, y, no_clusters, local_svm):
    test_index = np.unique(test_labels)
    local_predictions = []
    local_index = []
    for p in range(0, no_clusters):
        test_class = X_test[test_labels == p]
        local_index.append(test_class.index)
        svm = local_svm[p]
        test_y = y[test_labels == p]
        if len(test_class) > 1:
            test_class = std_list[p].transform(test_class)
            local_predictions.append(svm.predict(test_class))
            
        elif len(test_class) == 1:
            test_class = std_list[p].transform(test_class)
            print('Only one sample in cluster %.0f' % p)
            local_predictions.append(svm.predict(test_class))
            
        else:
            print('No samples in cluster %.0f' % p)
    
    local_predictions_df = pd.DataFrame(np.concatenate(local_predictions))
    local_index = np.concatenate(local_index)
    local_predictions_df.index = local_index
    return local_predictions_df