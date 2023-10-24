# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:01:25 2023

@author: Elise Lunde Gjelsvik
elise.lunde.gjelsvik@nmbu.no
"""
#Loading imports and data
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import SimpleRNN
# From functions file
from HCDL_functions import explained_variance_pls, remove_small_clusters, remove_small_clusters_test
from HCDL_functions import plsr_local_predictions, cnn_local_prediction, rnn_local_prediction, svm_local_predictions
from HCDL_functions import emsc

df_ftir = scipy.io.loadmat('FTIR_AMW.mat')
df_ftir_prep = scipy.io.loadmat('FTIR_AMW_tidied2.mat')

waves = np.logical_and(df_ftir['waves'] > 700, df_ftir['waves'] < 1800).flatten()
X = df_ftir['spectra']#[:, waves]
y = df_ftir['AMWall']
wave = df_ftir['waves'][waves]

rep = df_ftir_prep['replicates']
rep_nr = pd.DataFrame(rep)[0]

df = pd.DataFrame(X)
df.index = rep_nr

y_target = pd.Series(y[:, 0])
y_target.index = rep_nr

df_ftir['material'] -= 1
material = df_ftir['material']
enzyme_name = df_ftir['materName']

material2 = pd.DataFrame(material)
material2.index = rep_nr

# =============================================================================
# Average of spectra
# =============================================================================
# Combining the parallels and calculating the average
def average_calc(df):
    unique_id = np.unique(df.index)
    
    avg_list = []
    for i in range(len(unique_id)):
        avg = df[df.index == unique_id[i]]
        avg = avg.mean(axis=0) 
        avg_list.append(avg)
    
    avg_df = pd.concat(avg_list, axis=1, keys=[s.name for s in avg_list]).T
    avg_df.index = unique_id
    return avg_df

df = average_calc(df)
y_target = average_calc(pd.DataFrame(y_target))
sample_class = average_calc(pd.DataFrame(material2)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y_target, test_size=0.5, random_state=1, stratify=sample_class)
train_index = X_train.index
test_index = X_test.index

train_sample_class = sample_class[0][X_train.index]

X_cal, X_val, y_cal, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1, stratify=train_sample_class)
cal_index = X_cal.index
val_index = X_val.index

# =============================================================================
# EMSC preprosessering
# and dividing data in training/test/calibration/validation sets
# =============================================================================
X_train = np.array(X_train)
X_train = signal.savgol_filter(X_train, window_length=11, polyorder=3, deriv=2)

ref_spec = np.mean(X_train, axis=0) # Get a low baseline spectrum
X_train = emsc(X_train, ref_spec, d=2)

X_train = X_train[:, waves]
X_train = pd.DataFrame(X_train)
X_train.index = train_index

# =============================================================================
X_test = np.array(X_test)
X_test = signal.savgol_filter(X_test, window_length=11, polyorder=3, deriv=2)
X_test = emsc(X_test, ref_spec, d=2)

X_test = X_test[:, waves]
X_test = pd.DataFrame(X_test)
X_test.index = test_index

# =============================================================================
X_cal = np.array(X_cal)
X_cal = signal.savgol_filter(X_cal, window_length=11, polyorder=3, deriv=2)
X_cal = emsc(X_cal, ref_spec, d=2)

X_cal = X_cal[:, waves]
X_cal = pd.DataFrame(X_cal)
X_cal.index = cal_index

# =============================================================================
X_val = np.array(X_val)
X_val = signal.savgol_filter(X_val, window_length=11, polyorder=3, deriv=2)
X_val = emsc(X_val, ref_spec, d=2)

X_val = X_val[:, waves]
X_val = pd.DataFrame(X_val)
X_val.index = val_index

# =============================================================================
# Global PLSR
# =============================================================================
index_list = np.unique(X_train.index)            

global_mse = []
score_mse = []
for i in range(len(index_list)):
    validation = X_train[X_train.index == index_list[i]]
    training = X_train.drop([index_list[i]], axis=0)
        
    validation_y = y_train[y_train.index == index_list[i]]
    training_y = y_train.drop([index_list[i]], axis=0)
    
    std_X = StandardScaler(with_mean=True, with_std=False)
    training = std_X.fit_transform(training)
    validation = std_X.transform(validation)
    
    score_comp = []
    for j in range(1, 30):
        pls = PLSRegression(n_components=j, scale=False)
        pls.fit(training, training_y)        
        
        y_pred = pls.predict(validation)
        mod_score = mean_squared_error(validation_y, y_pred)
        score_comp.append(mod_score)
    
    score_mse.append(score_comp)
    
score_mse_df = np.stack((score_mse), axis=0)
score_mse_mean = np.mean(score_mse_df, axis=0)
global_n = score_mse_mean.argmin() + 1

# =============================================================================
# Adding restraint that each included components should account for more than 1% 
# of the explained variance
pls = PLSRegression(n_components=global_n, scale=False)

std_X = StandardScaler(with_mean=True, with_std=False)
X_train_std = std_X.fit_transform(X_train)
pls.fit(X_train_std, y_train)

expl_var = explained_variance_pls(pls, X_train_std)
for k in range(len(expl_var)):
    if expl_var[k] / sum(expl_var) > 0.01:
       global_n = k + 1
    else:
        break

# Fitting the global model with the determined #components
pls = PLSRegression(n_components=global_n, scale=False)    
pls.fit(X_train_std, y_train)
X_scores = pls.transform(X_train_std)    

# =============================================================================
# HC-PLSR
# =============================================================================
no_clusters = 2

fcm = FCM(n_clusters=no_clusters, random_state=12)
fcm.fit(X_scores)

fcm_labels = remove_small_clusters(fcm, X_train, no_clusters)[0]
label_dict = remove_small_clusters(fcm, X_train, no_clusters)[1]

print(Counter(fcm_labels))

# =============================================================================
local_score = []
for k in range(0, no_clusters):
    cluster_k = X_train[fcm_labels == k]
    y_k = y_train[fcm_labels == k]
        
    if len(cluster_k) > 0:
        local_mse = []
        interm_pls = []
        index_k = np.unique(cluster_k.index)
        for l in range(len(index_k)):
            validation = cluster_k[cluster_k.index == index_k[l]]
            training = cluster_k.drop([index_k[l]], axis=0)
            
            validation_y = y_k[y_k.index == index_k[l]]
            training_y = y_k.drop([index_k[l]], axis=0)
            
            std_X = StandardScaler(with_mean=True, with_std=False)
            training = std_X.fit_transform(training)
            validation = std_X.transform(validation)
            
            score_comp = []
            for n in range(1, len(training)):
                pls = PLSRegression(n_components=n, scale=False)
                pls.fit(training, training_y)
                y_pred = pls.predict(validation)
                mod_score = mean_squared_error(validation_y, y_pred)
                score_comp.append(mod_score)
            
            local_mse.append(score_comp)
        
        score_ = np.stack((local_mse), axis=0)
        score_mean = np.mean(score_, axis=0)
        local_score.append(score_mean)        

    else:
        print('No samples in cluster %.0f' % k)
        local_score.append([])   
        
local_components = []
for m in range(len(local_score)):
    if len(local_score[m]) > 0:
        local_components.append(local_score[m].argmin()+1)
    else:
        local_components.append(0)    

local_pls = [] 
std_list = []
for q in range(len(local_components)):
    pls = PLSRegression(n_components=local_components[q], scale=False)

    cluster_q = X_train[fcm_labels == q]
    y_q = y_train[fcm_labels == q]
    
    if len(cluster_q) > 0:
        std_X = StandardScaler(with_mean=True, with_std=False)
        cluster_q = std_X.fit_transform(cluster_q)
    
        pls.fit(cluster_q, y_q)
        local_pls.append(pls)
        std_list.append(std_X)
    else:
        local_pls.append(pls)
        std_list.append(0)

# =============================================================================
# Classification and prediction
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_train_std = std_X.fit_transform(X_train)
global_model.fit(X_train_std, y_train)
X_scores_plsr = global_model.transform(X_train_std)

X_test_std = std_X.transform(X_test)
test_scores = global_model.transform(X_test_std)

test_fcm = remove_small_clusters_test(X_test, test_scores, fcm, label_dict, fcm_labels, no_clusters)

fcm_prediction = plsr_local_predictions(std_list, test_fcm, X_test, y_test, no_clusters, local_pls)
hc_plsr_fcm = round(r2_score(y_test.sort_index(), fcm_prediction.sort_index()), 3)
print('R2 score for HC-PLSR with %.0f clusters: %.3f' % (no_clusters, hc_plsr_fcm))

# =============================================================================
lda = LDA()
lda.fit(X_scores_plsr, fcm_labels)
test_lda = lda.predict(test_scores)

lda_prediction = plsr_local_predictions(std_list, test_lda, X_test, y_test, no_clusters, local_pls)
hc_plsr_lda = round(r2_score(y_test.sort_index(), lda_prediction.sort_index()), 3)
print('R2 score for HC-PLSR with %.0f clusters: %.3f' % (no_clusters, hc_plsr_lda))

# =============================================================================
qda = QDA()
qda.fit(X_scores_plsr, fcm_labels)
test_qda = qda.predict(test_scores)

qda_prediction = plsr_local_predictions(std_list, test_qda, X_test, y_test, no_clusters, local_pls)
hc_plsr_qda = round(r2_score(y_test.sort_index(), qda_prediction.sort_index()), 3)
print('R2 score for HC-PLSR with %.0f clusters: %.3f' % (no_clusters, hc_plsr_qda))

# =============================================================================
gnb = GaussianNB()
gnb.fit(X_scores_plsr, fcm_labels)
test_gnb = gnb.predict(test_scores)

gnb_prediction = plsr_local_predictions(std_list, test_gnb, X_test, y_test, no_clusters, local_pls)
hc_plsr_gnb = round(r2_score(y_test.sort_index(), gnb_prediction.sort_index()), 3)
print('R2 score for HC-PLSR with %.0f clusters: %.3f' % (no_clusters, hc_plsr_gnb))

# =============================================================================
y_pred = global_model.predict(X_test_std)
global_test = r2_score(y_test, y_pred)
print('R2 score for global PLSR model: %.3f' % global_test)

# =============================================================================
# Results
# =============================================================================
r2_list = [hc_plsr_fcm, hc_plsr_lda, hc_plsr_qda, hc_plsr_gnb, global_test]
r2_list_pls = np.array(r2_list)

# pickle.dump(r2_list_pls, open(save_folder + "/r2_pls.p", "wb")) 

# =============================================================================
# Finding the optimal no. of clusters PLSR
# =============================================================================
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_cal_std = std_X.fit_transform(X_cal)
global_model.fit(X_cal_std, y_cal)
cal_scores = global_model.transform(X_cal_std)

X_val_std = std_X.transform(X_val)
val_scores = global_model.transform(X_val_std)

y_pred = global_model.predict(X_val_std)
global_val = r2_score(y_val, y_pred)
print('R2 calibration-score for global PLSR model: %.3f' % global_val)

opt_cluster_pls = []
for i in range(2, 11):
    fcm = FCM(n_clusters=i, random_state=12)
    fcm.fit(cal_scores)
    
    fcm_labels = remove_small_clusters(fcm, X_cal, i)[0]
    label_dict = remove_small_clusters(fcm, X_cal, i)[1]
    
    local_score = []
    for k in range(0, i):
        cluster_k = X_cal[fcm_labels == k]
        y_k = y_cal[fcm_labels == k]
        
        if len(cluster_k) > 0:
            local_mse = []
            interm_pls = []
            index_k = np.unique(cluster_k.index)
            for l in range(len(index_k)):
                validation = cluster_k[cluster_k.index == index_k[l]]
                training = cluster_k.drop([index_k[l]], axis=0)
                
                validation_y = y_k[y_k.index == index_k[l]]
                training_y = y_k.drop([index_k[l]], axis=0)
                
                #Standardiser her
                std_X = StandardScaler(with_mean=True, with_std=False)
                training = std_X.fit_transform(training)
                validation = std_X.transform(validation)
            
                score_comp = []
                for n in range(1, len(training)):
                    pls = PLSRegression(n_components=n, scale=False)
                    pls.fit(training, training_y)
                    y_pred = pls.predict(validation)
                    mod_score = mean_squared_error(validation_y, y_pred)
                    score_comp.append(mod_score)
                
                local_mse.append(score_comp)
                
            score_ = np.stack((local_mse), axis=0)
            score_mean = np.mean(score_, axis=0)
            local_score.append(score_mean)
                
        else:
            print('No samples in cluster %.0f' % k)
            local_score.append([])            
                
    local_components = []
    for m in range(len(local_score)):
        if len(local_score[m]) > 0:
            local_components.append(local_score[m].argmin()+1)
        else:
            local_components.append(0)

    local_pls = [] 
    std_list = []
    # training_check = []   
    for q in range(len(local_components)):
        pls = PLSRegression(n_components=local_components[q], scale=False)
        
        cluster_q = X_cal[fcm_labels == q]
        y_q = y_cal[fcm_labels == q]
        
        if len(cluster_q) > 0:
            std_X = StandardScaler(with_mean=True, with_std=False)
            cluster_q = std_X.fit_transform(cluster_q)
        
            pls.fit(cluster_q, y_q)
            local_pls.append(pls)
            std_list.append(std_X)
            # training_check.append(cluster_q.index)
        else:
            local_pls.append(pls)
            std_list.append(0)

    test_fcm = remove_small_clusters_test(X_val, val_scores, fcm, label_dict, fcm_labels, i)
    fcm_prediction = plsr_local_predictions(std_list, test_fcm, X_val, y_val, i, local_pls)
    hc_plsr_fcm = r2_score(y_val.sort_index(), fcm_prediction.sort_index())
    print('R2 score for HC-PLSR with %.0f clusters: %.2f' % (i, hc_plsr_fcm))

    lda = LDA()
    lda.fit(cal_scores, fcm_labels)
    test_lda = lda.predict(val_scores)

    lda_prediction = plsr_local_predictions(std_list, test_lda, X_val, y_val, i, local_pls)
    hc_plsr_lda = r2_score(y_val.sort_index(), lda_prediction.sort_index())
    print('R2 score for HC-PLSR with %.0f clusters: %.2f' % (i, hc_plsr_lda))

    qda = QDA()
    qda.fit(cal_scores, fcm_labels)
    test_qda = qda.predict(val_scores)

    qda_prediction = plsr_local_predictions(std_list, test_qda, X_val, y_val, i, local_pls)
    hc_plsr_qda = r2_score(y_val.sort_index(), qda_prediction.sort_index())
    print('R2 score for HC-PLSR with %.0f clusters: %.2f' % (i, hc_plsr_qda))

    gnb = GaussianNB()
    gnb.fit(cal_scores, fcm_labels)
    test_gnb = gnb.predict(val_scores)

    gnb_prediction = plsr_local_predictions(std_list, test_gnb, X_val, y_val, i, local_pls)
    hc_plsr_gnb = r2_score(y_val.sort_index(), gnb_prediction.sort_index())
    print('R2 score for HC-PLSR with %.0f clusters: %.2f' % (i, hc_plsr_gnb))

    r2_list = [hc_plsr_fcm, hc_plsr_lda, hc_plsr_qda, hc_plsr_gnb, global_val]
    r2_list = np.array(r2_list)
    
    opt_cluster_pls.append(r2_list)
    
# =============================================================================
# CNN global model
# =============================================================================
def build_network():
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=11, activation='elu', input_shape=(X_train_centered.shape[1:])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    #model.summary()
    
    #opt = SGD(learning_rate=0.00001)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

e = 1000
index_list = np.unique(X_train.index) 

global_score = []
for m in range(len(X_train)):
    global_validation = X_train[X_train.index == index_list[m]]
    global_training = X_train.drop([index_list[m]], axis=0)
    
    global_validation_y = y_train[y_train.index == index_list[m]]
    global_training_y = y_train.drop([index_list[m]], axis=0)   
    
    std_X = StandardScaler(with_mean=True, with_std=True)
    training = std_X.fit_transform(global_training)
    validation = std_X.transform(global_validation)

    n_features = 1
    X_train_centered = np.array(training)
    X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], X_train_centered.shape[1], n_features))
    X_valid = np.array(validation)
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], n_features))     

    tf.keras.backend.clear_session() 
    model = build_network()
    history = model.fit(X_train_centered, global_training_y, batch_size=8, epochs=e, 
                        verbose=0, shuffle=True) 
    y_pred = model.predict(X_valid)

    mod_score = mean_squared_error(global_validation_y, y_pred)
    score_comp = history.history['loss'][::10]
    
    score_ = np.stack((score_comp), axis=0)
    global_score.append(score_)
    
global_score2 = np.array(global_score) 
global_score2 = np.mean(global_score2, axis=0)   
low_score = global_score2.argmin()
ep_list = pd.DataFrame([*range(1, e, 10)])
global_epoch = ep_list.loc[low_score][0]

std_X = StandardScaler(with_mean=True, with_std=True)
training = std_X.fit_transform(X_train)
validation = std_X.transform(X_test)

n_features = 1
X_train_centered = np.array(training)
X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], X_train_centered.shape[1], n_features))
X_valid = np.array(validation)
X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], n_features))     

model = build_network()
history = model.fit(X_train_centered, y_train, batch_size=8, epochs=global_epoch, 
                    verbose=0, shuffle=True) 

y_pred = model.predict(X_valid)
global_cnn = r2_score(y_test, y_pred)    

# =============================================================================
no_clusters = 2

fcm = FCM(n_clusters=no_clusters, random_state=12)
fcm.fit(X_scores)

fcm_labels = remove_small_clusters(fcm, X_train, no_clusters)[0]
label_dict = remove_small_clusters(fcm, X_train, no_clusters)[1]

print(Counter(fcm_labels))

# =============================================================================
# HC-CNN
# =============================================================================
e = 1000

local_score = []
for k in range(0, no_clusters):
    cluster_k = X_train[fcm_labels == k]
    y_k = y_train[fcm_labels == k]
    
    if len(cluster_k) > 0:
        local_mse = []
        cnn_prediction = []
        index_k = np.unique(cluster_k.index)
        for l in range(len(index_k)):
            validation = cluster_k[cluster_k.index == index_k[l]]
            training = cluster_k.drop([index_k[l]], axis=0)
            
            validation_y = y_k[y_k.index == index_k[l]]
            training_y = y_k.drop([index_k[l]], axis=0)
            
            #Standardiser her
            std_X = StandardScaler(with_mean=True, with_std=True)
            training = std_X.fit_transform(training)
            validation = std_X.transform(validation)
            
            n_features = 1
            X_train_centered = np.array(training)
            X_valid = np.array(validation)
            X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], X_train_centered.shape[1], n_features))
            X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], n_features))        
            
            tf.keras.backend.clear_session()            
            model = build_network()
            history = model.fit(X_train_centered, training_y, batch_size=8, epochs=e, 
                                verbose=0, shuffle=True)
                
            y_pred = model.predict(X_valid)
            mod_score = mean_squared_error(validation_y, y_pred)
            score_comp = history.history['loss'][::10]
                        
            local_mse.append(score_comp)
            
        score_ = np.stack((local_mse), axis=0)
        score_mean = np.mean(score_, axis=0)
        local_score.append(score_mean)
        
    else: 
        print('No samples in cluster %.0f' % k)
        local_score.append([]) 
    
    
local_components = []
for m in range(len(local_score)):
    if len(local_score[m]) > 0:
        low_score = local_score[m].argmin()
        ep_list = pd.DataFrame([*range(1, e, 10)])
        low_ep = ep_list.loc[low_score][0]
        local_components.append(low_ep)
    else:
        local_components.append(0)

local_cnn = [] 
training_check = []
std_list = []   
for q in range(len(local_components)):
    model = build_network()

    cluster_q = X_train[fcm_labels == q]
    y_q = y_train[fcm_labels == q]
    training_check.append(cluster_q.index)
    
    if len(cluster_q) > 0:
        std_X = StandardScaler(with_mean=True, with_std=True)
        cluster_q = std_X.fit_transform(cluster_q)
    
        n_features = 1
        X_train_centered = np.array(cluster_q)
        X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], X_train_centered.shape[1], n_features))
        
        history = model.fit(X_train_centered, y_q, batch_size=8, epochs=local_components[q],
                            verbose=0, shuffle=True)
        
        local_cnn.append(model)
        std_list.append(std_X)
    else:
        local_cnn.append(model)
        std_list.append(0)

# =============================================================================
# Classification and prediction
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_train_std = std_X.fit_transform(X_train)
global_model.fit(X_train_std, y_train)
X_scores_cnn = global_model.transform(X_train_std)

X_test_std = std_X.transform(X_test)
test_scores = global_model.transform(X_test_std)

test_fcm = remove_small_clusters_test(X_test, test_scores, fcm, label_dict, fcm_labels, no_clusters)
fcm_prediction = cnn_local_prediction(std_list, X_test, test_fcm, y_test, no_clusters, local_cnn)
hc_cnn_fcm = r2_score(y_test.sort_index(), fcm_prediction.sort_index())
print('R2 score for HC-CNN with %.0f clusters: %.2f' % (no_clusters, hc_cnn_fcm))

# =============================================================================
lda = LDA()
lda.fit(X_scores_cnn, fcm_labels)
test_lda = lda.predict(test_scores)

lda_prediction = cnn_local_prediction(std_list, X_test, test_lda, y_test, no_clusters, local_cnn)
hc_cnn_lda = r2_score(y_test.sort_index(), lda_prediction.sort_index())
print('R2 score for HC-CNN with %.0f clusters: %.2f' % (no_clusters, hc_cnn_lda))

# =============================================================================
qda = QDA()
qda.fit(X_scores_cnn, fcm_labels)
test_qda = qda.predict(test_scores)

qda_prediction = cnn_local_prediction(std_list, X_test, test_qda, y_test, no_clusters, local_cnn)
hc_cnn_qda = r2_score(y_test.sort_index(), qda_prediction.sort_index())
print('R2 score for HC-CNN with %.0f clusters: %.2f' % (no_clusters, hc_cnn_qda))

# =============================================================================
gnb = GaussianNB()
gnb.fit(X_scores_cnn, fcm_labels)
test_gnb = gnb.predict(test_scores)

gnb_prediction = cnn_local_prediction(std_list, X_test, test_gnb, y_test, no_clusters, local_cnn)
hc_cnn_gnb = r2_score(y_test.sort_index(), gnb_prediction.sort_index())
print('R2 score for HC-CNN with %.0f clusters: %.2f' % (no_clusters, hc_cnn_gnb))

# =============================================================================
y_pred = global_model.predict(X_test_std)
global_test_cnn = r2_score(y_test, y_pred)
print('R2 score for global PLSR model: %.2f' % global_test_cnn)

# =============================================================================
# Results
# =============================================================================
r2_list_cnn = [hc_cnn_fcm, hc_cnn_lda, hc_cnn_qda, hc_cnn_gnb, global_test_cnn]
r2_list_cnn = np.array(r2_list_cnn)

# =============================================================================
# Important features
# =============================================================================
def vargrad_input(blocks_train, network, type="max", ensemble_num=20, quantile=0.9, seed=1):
    np_random_gen = np.random.default_rng(seed)
    all_grads = [[] for _ in range(len(blocks_train))]
    out = []
    for j in range(ensemble_num):
        noised_data_blocks = [data_block + np_random_gen.normal(loc=0, scale=1, size=(np.shape(data_block)[0],
                                                                np.shape(data_block)[1])) for data_block in blocks_train]

        input_tensors = tf.convert_to_tensor(noised_data_blocks)
        with tf.GradientTape() as tape:
            tape.watch(input_tensors)
            output = network(input_tensors)
            positive = tf.reduce_sum(output, axis=1)
        positive_grads = tape.gradient(positive, input_tensors)
        positive_grads = [grad.numpy() for grad in positive_grads]

        for i in range(len(blocks_train)):
            all_grads[i].append(np.squeeze(positive_grads[i]))
        out.append(output)

    vargrad = [np.array(all_grad).std(axis=0) for all_grad in all_grads]
    vargrad_df = np.squeeze(vargrad).T

    if type=="max":
        return [grad.max() for grad in vargrad_df]
    elif type=="mean":
        return [grad.mean() for grad in vargrad_df]
    elif type=="sum":
        return [grad.sum() for grad in vargrad_df]
    elif type=="quantile":
        return [np.quantile(grad, quantile) for grad in vargrad_df]
    
def visualizing_local_models(local_models, training_data, labels, std_list, net='cnn'):
    vargrad_res = []
    
    for c in range(len(local_models)):
        model = local_models[c]
        X_cluster = training_data[labels == c]
        X_train_c = std_list[c].transform(X_cluster)  
        
        n_features = 1
        X_train_c = np.array(X_train_c)
        if net=='cnn':
            X_train_c = X_train_c.reshape((X_train_c.shape[0], X_train_c.shape[1], n_features))
        elif net=='rnn':
            X_train_c = X_train_c.reshape((X_train_c.shape[0], n_features, X_train_c.shape[1]))
            
        feature_gradient = vargrad_input(X_train_c, model)
        vargrad_res.append(feature_gradient)
        
    return vargrad_res

cnn_feature_vis = visualizing_local_models(local_cnn, X_train, fcm_labels, std_list, net='cnn')

# =============================================================================
# Finding the optimal numbers of clusters for HC-CNN
# =============================================================================
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_cal_std = std_X.fit_transform(X_cal)
global_model.fit(X_cal_std, y_cal)
cal_scores = global_model.transform(X_cal_std)

X_val_std = std_X.transform(X_val)
val_scores = global_model.transform(X_val_std)

y_pred = global_model.predict(X_val_std)
global_val = r2_score(y_val, y_pred)
print('R2 calibration-score for global PLSR model: %.3f' % global_val)

opt_cluster_cnn = []
for i in range(2, 11):
    fcm = FCM(n_clusters=i, random_state=12)
    fcm.fit(cal_scores)
    
    fcm_labels = remove_small_clusters(fcm, X_cal, i)[0]
    label_dict = remove_small_clusters(fcm, X_cal, i)[1]

    local_score = []
    for k in range(0, i):
        cluster_k = X_cal[fcm_labels == k]
        y_k = y_cal[fcm_labels == k]
    
        if len(cluster_k) > 0:
            local_mse = []
            cnn_prediction = []
            index_k = np.unique(cluster_k.index)
            for l in range(len(index_k)):
                validation = cluster_k[cluster_k.index == index_k[l]]
                training = cluster_k.drop([index_k[l]], axis=0)
                
                validation_y = y_k[y_k.index == index_k[l]]
                training_y = y_k.drop([index_k[l]], axis=0)
                
                std_X = StandardScaler(with_mean=True, with_std=True)
                training = std_X.fit_transform(training)
                validation = std_X.transform(validation)
                
                n_features = 1
                X_train_centered = np.array(training)
                X_valid = np.array(validation)
                X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], X_train_centered.shape[1], n_features))
                X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], n_features))        
                
                tf.keras.backend.clear_session() 
                model = build_network()
                history = model.fit(X_train_centered, training_y, batch_size=8, epochs=e, 
                                    verbose=0, shuffle=True, validation_data=(X_valid, validation_y))
                
                y_pred = model.predict(X_valid)
                mod_score = mean_squared_error(validation_y, y_pred)
                score_comp = history.history['loss'][::10]
                        
                local_mse.append(score_comp)
        
            score_ = np.stack((local_mse), axis=0)
            score_mean = np.mean(score_, axis=0)
            local_score.append(score_mean)
        else:
            print('No samples in cluster %.0f' % k)
            local_score.append([]) 
                
    local_components = []
    for m in range(len(local_score)):
        if len(local_score[m]):
            low_score = local_score[m].argmin()
            ep_list = pd.DataFrame([*range(1, e, 10)])
            low_ep = ep_list.loc[low_score][0]
            local_components.append(low_ep)
        else: 
            local_components.append(0)

    local_cnn = [] 
    std_list = []     
    for q in range(len(local_components)):
        model = build_network()
    
        cluster_q = X_cal[fcm_labels == q]
        y_q = y_cal[fcm_labels == q]
        
        if len(cluster_q) > 0:
            std_X = StandardScaler(with_mean=True, with_std=True)
            cluster_q = std_X.fit_transform(cluster_q)
        
            n_features = 1
            X_train_centered = np.array(cluster_q)
            X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], X_train_centered.shape[1], n_features))
        
            history = model.fit(X_train_centered, y_q, batch_size=8, epochs=local_components[q],
                                verbose=0, shuffle=True)
    
            local_cnn.append(model)
            std_list.append(std_X)
        else:
            local_cnn.append(model)
            std_list.append(0)

    test_fcm = remove_small_clusters_test(X_val, val_scores, fcm, label_dict, fcm_labels, i)
    fcm_prediction = cnn_local_prediction(std_list, X_val, test_fcm, y_val, i, local_cnn)
    hc_cnn_fcm = r2_score(y_val.sort_index(), fcm_prediction.sort_index())
    print('R2 score for HC-CNN with %.0f clusters: %.2f' % (i, hc_cnn_fcm))

    lda = LDA()
    lda.fit(cal_scores, fcm_labels)
    test_lda = lda.predict(val_scores)

    lda_prediction = cnn_local_prediction(std_list, X_val, test_lda, y_val, i, local_cnn)
    hc_cnn_lda = r2_score(y_val.sort_index(), lda_prediction.sort_index())
    print('R2 score for HC-CNN with %.0f clusters: %.2f' % (i, hc_cnn_lda))

    qda = QDA()
    qda.fit(cal_scores, fcm_labels)
    test_qda = qda.predict(val_scores)

    qda_prediction = cnn_local_prediction(std_list, X_val, test_qda, y_val, i, local_cnn)
    hc_cnn_qda = r2_score(y_val.sort_index(), qda_prediction.sort_index())
    print('R2 score for HC-CNN with %.0f clusters: %.2f' % (i, hc_cnn_qda))

    gnb = GaussianNB()
    gnb.fit(cal_scores, fcm_labels)
    test_gnb = gnb.predict(val_scores)

    gnb_prediction = cnn_local_prediction(std_list, X_val, test_gnb, y_val, i, local_cnn)
    hc_cnn_gnb = r2_score(y_val.sort_index(), gnb_prediction.sort_index())
    print('R2 score for HC-CNN with %.0f clusters: %.2f' % (i, hc_cnn_gnb))

    r2_list = [hc_cnn_fcm, hc_cnn_lda, hc_cnn_qda, hc_cnn_gnb, global_val]
    r2_list = np.array(r2_list)
    
    opt_cluster_cnn.append(r2_list)  
    
# =============================================================================
# RNN
# =============================================================================
def build_rnn():
    model = Sequential()
    model.add(SimpleRNN(32, input_shape=X_train_centered.shape[1:], return_sequences=True, activation='elu'))
    model.add(SimpleRNN(16, activation='elu'))
    model.add(Dense(8))
    model.add(Dense(1, activation='linear'))
    
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

global_score = []
for m in range(len(X_train)):
    global_validation = X_train[X_train.index == index_list[m]]
    global_training = X_train.drop([index_list[m]], axis=0)
    
    global_validation_y = y_train[y_train.index == index_list[m]]
    global_training_y = y_train.drop([index_list[m]], axis=0)   
    
    std_X = StandardScaler(with_mean=True, with_std=True)
    training = std_X.fit_transform(global_training)
    validation = std_X.transform(global_validation)

    n_features = 1
    X_train_centered = np.array(training)
    X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], n_features, X_train_centered.shape[1]))
    X_valid = np.array(validation)
    X_valid = X_valid.reshape((X_valid.shape[0], n_features, X_valid.shape[1]))     

    tf.keras.backend.clear_session() 
    model = build_rnn()
    history = model.fit(X_train_centered, global_training_y, batch_size=8, epochs=e,
                        verbose=0)
    y_pred = model.predict(X_valid)

    mod_score = mean_squared_error(global_validation_y, y_pred)
    score_comp = history.history['loss'][::10]
    
    score_ = np.stack((score_comp), axis=0)
    global_score.append(score_)
    
global_score2 = np.array(global_score)  
global_score2 = np.mean(global_score2, axis=0)   
low_score = global_score2.argmin()
ep_list = pd.DataFrame([*range(1, e, 10)])
global_epoch = ep_list.loc[low_score][0]

std_X = StandardScaler(with_mean=True, with_std=True)
training = std_X.fit_transform(X_train)
validation = std_X.transform(X_test)

n_features = 1
X_train_centered = np.array(training)
X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], n_features, X_train_centered.shape[1]))
X_valid = np.array(validation)
X_valid = X_valid.reshape((X_valid.shape[0], n_features, X_valid.shape[1]))    

model = build_rnn()
history = model.fit(X_train_centered, y_train, batch_size=8, epochs=e,
                    verbose=0)

y_pred = model.predict(X_valid)
global_rnn = r2_score(y_test, y_pred)

# =============================================================================
no_clusters = 2

fcm = FCM(n_clusters=no_clusters, random_state=12)
fcm.fit(X_scores)

fcm_labels = remove_small_clusters(fcm, X_train, no_clusters)[0]
label_dict = remove_small_clusters(fcm, X_train, no_clusters)[1]


print(Counter(fcm_labels))

# =============================================================================
e = 1000

local_score = []
for k in range(0, no_clusters):
    cluster_k = X_train[fcm_labels == k]
    y_k = y_train[fcm_labels == k]
    
    if len(cluster_k) > 0:
        local_mse = []
        rnn_prediction = []
        index_k = np.unique(cluster_k.index)
        for l in range(len(index_k)):
            validation = cluster_k[cluster_k.index == index_k[l]]
            training = cluster_k.drop([index_k[l]], axis=0)
            
            validation_y = y_k[y_k.index == index_k[l]]
            training_y = y_k.drop([index_k[l]], axis=0)
            
            std_X = StandardScaler(with_mean=True, with_std=True)
            training = std_X.fit_transform(training)
            validation = std_X.transform(validation)
            
            n_features = 1
            X_train_centered = np.array(training)
            X_valid = np.array(validation)
            X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], n_features, X_train_centered.shape[1]))
            X_valid = X_valid.reshape((X_valid.shape[0], n_features, X_valid.shape[1]))        

            tf.keras.backend.clear_session()            
            model = build_rnn()
            history = model.fit(X_train_centered, training_y, batch_size=8, epochs=e, 
                                verbose=0, shuffle=True, validation_data=(X_valid, validation_y))
                
            y_pred = model.predict(X_valid)
            mod_score = mean_squared_error(validation_y, y_pred)
            score_comp = history.history['loss'][::10]
                
            local_mse.append(score_comp)
        
        score_ = np.stack((local_mse), axis=0)
        score_mean = np.mean(score_, axis=0)
        local_score.append(score_mean)
    else:
        print('No samples in cluster %.0f' % k)
        local_score.append([])  
    
local_components = []
for m in range(len(local_score)):
    if len(local_score[m]) > 0:
        low_score = local_score[m].argmin()
        ep_list = pd.DataFrame([*range(1, e, 10)])
        low_ep = ep_list.loc[low_score][0]
        local_components.append(low_ep)
    else:
        local_components.append(0)

local_rnn = []
std_list = [] 
for q in range(len(local_components)):
    model = build_rnn()

    cluster_q = X_train[fcm_labels == q]
    y_q = y_train[fcm_labels == q]
    
    if len(cluster_q) > 0:
        std_X = StandardScaler(with_mean=True, with_std=True)
        cluster_q = std_X.fit_transform(cluster_q)
        
        n_features = 1
        X_train_centered = np.array(cluster_q)
        X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], n_features, X_train_centered.shape[1]))
        
        history = model.fit(X_train_centered, y_q, batch_size=8, epochs=local_components[q],
                            verbose=0, shuffle=True)

        local_rnn.append(model)
        std_list.append(std_X)
    else:
        local_rnn.append(model)
        std_list.append(0)
    
# =============================================================================
# Classification and prediction
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_train_std = std_X.fit_transform(X_train)
global_model.fit(X_train_std, y_train)
X_scores_rnn = global_model.transform(X_train_std)

X_test_std = std_X.transform(X_test)
test_scores = global_model.transform(X_test_std)

test_fcm = remove_small_clusters_test(X_test, test_scores, fcm, label_dict, fcm_labels, no_clusters)
fcm_prediction = rnn_local_prediction(std_list, X_test, test_fcm, y_test, no_clusters, local_rnn)
hc_rnn_fcm = r2_score(y_test.sort_index(), fcm_prediction.sort_index())
print('R2 score for HC-RNN with %.0f clusters: %.2f' % (no_clusters, hc_rnn_fcm))

# =============================================================================
lda = LDA()
lda.fit(X_scores_rnn, fcm_labels)
test_lda = lda.predict(test_scores)

lda_prediction = rnn_local_prediction(std_list, X_test, test_lda, y_test, no_clusters, local_rnn)
hc_rnn_lda = r2_score(y_test.sort_index(), lda_prediction.sort_index())
print('R2 score for HC-RNN with %.0f clusters: %.2f' % (no_clusters, hc_rnn_lda))

# =============================================================================
qda = QDA()
qda.fit(X_scores_rnn, fcm_labels)
test_qda = qda.predict(test_scores)

qda_prediction = rnn_local_prediction(std_list, X_test, test_qda, y_test, no_clusters, local_rnn)
hc_rnn_qda = r2_score(y_test.sort_index(), qda_prediction.sort_index())
print('R2 score for HC-RNN with %.0f clusters: %.2f' % (no_clusters, hc_rnn_qda))

# =============================================================================
gnb = GaussianNB()
gnb.fit(X_scores_rnn, fcm_labels)
test_gnb = gnb.predict(test_scores)

gnb_prediction = rnn_local_prediction(std_list, X_test, test_gnb, y_test, no_clusters, local_rnn)
hc_rnn_gnb = r2_score(y_test.sort_index(), gnb_prediction.sort_index())
print('R2 score for HC-RNN with %.0f clusters: %.2f' % (no_clusters, hc_rnn_gnb))

# =============================================================================
y_pred = global_model.predict(X_test_std)
global_test_rnn = r2_score(y_test, y_pred)
print('R2 score for global PLSR model: %.2f' % global_test_rnn) 

# =============================================================================
# Results
# =============================================================================
r2_list_rnn = [hc_rnn_fcm, hc_rnn_lda, hc_rnn_qda, hc_rnn_gnb, global_test_rnn]
r2_list_rnn = np.array(r2_list_rnn)

# =============================================================================
rnn_feature_vis = visualizing_local_models(local_rnn, X_train, fcm_labels, std_list, net='rnn')

# =============================================================================
# Finding the optimal number of clusters for HC-RNN
# =============================================================================
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_cal_std = std_X.fit_transform(X_cal)
global_model.fit(X_cal_std, y_cal)
cal_scores = global_model.transform(X_cal_std)

X_val_std = std_X.transform(X_val)
val_scores = global_model.transform(X_val_std)

y_pred = global_model.predict(X_val_std)
global_val = r2_score(y_val, y_pred)
print('R2 calibration-score for global PLSR model: %.3f' % global_val)

opt_cluster_rnn = []
for i in range(2, 11):
    fcm = FCM(n_clusters=i, random_state=12)
    fcm.fit(cal_scores)
    
    fcm_labels = remove_small_clusters(fcm, X_cal, i)[0]
    label_dict = remove_small_clusters(fcm, X_cal, i)[1]

    local_score = []
    for k in range(0, i):
        cluster_k = X_cal[fcm_labels == k]
        y_k = y_cal[fcm_labels == k]
    
        if len(cluster_k) > 0:
            local_mse = []
            rnn_prediction = []
            index_k = np.unique(cluster_k.index)
            for l in range(len(index_k)):
                validation = cluster_k[cluster_k.index == index_k[l]]
                training = cluster_k.drop([index_k[l]], axis=0)
                
                validation_y = y_k[y_k.index == index_k[l]]
                training_y = y_k.drop([index_k[l]], axis=0)
                
                std_X = StandardScaler(with_mean=True, with_std=True)
                training = std_X.fit_transform(training)
                validation = std_X.transform(validation)
                
                n_features = 1
                X_train_centered = np.array(training)
                X_valid = np.array(validation)
                X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], n_features, X_train_centered.shape[1]))
                X_valid = X_valid.reshape((X_valid.shape[0], n_features, X_valid.shape[1]))        
                
                tf.keras.backend.clear_session() 
                model = build_rnn()
                history = model.fit(X_train_centered, training_y, batch_size=8, epochs=e, 
                                    verbose=0, shuffle=True)
            
                y_pred = model.predict(X_valid)
                mod_score = mean_squared_error(validation_y, y_pred)
                score_comp = history.history['loss'][::10]
                        
                local_mse.append(score_comp)
        
            score_ = np.stack((local_mse), axis=0)
            score_mean = np.mean(score_, axis=0)
            local_score.append(score_mean)
        else:
            print('No samples in cluster %.0f' % k)
            local_score.append([])
                
    local_components = []
    for m in range(len(local_score)):
        if len(local_score[m]) > 0:
            low_score = local_score[m].argmin()
            ep_list = pd.DataFrame([*range(1, e, 10)])
            low_ep = ep_list.loc[low_score][0]
            local_components.append(low_ep)
        else:
            local_components.append(0)
    
    local_rnn = [] 
    std_list = []     
    for q in range(len(local_components)):
        model = build_rnn()
    
        cluster_q = X_cal[fcm_labels == q]
        y_q = y_cal[fcm_labels == q]
        
        if len(cluster_q) > 0:
            std_X = StandardScaler(with_mean=True, with_std=True)
            cluster_q = std_X.fit_transform(cluster_q)
        
            n_features = 1
            X_train_centered = np.array(cluster_q)
            X_train_centered = X_train_centered.reshape((X_train_centered.shape[0], n_features, X_train_centered.shape[1]))
            
            history = model.fit(X_train_centered, y_q, batch_size=8, epochs=local_components[q],
                                verbose=0, shuffle=True)
            
            local_rnn.append(model)
            std_list.append(std_X)
        else:
            local_rnn.append(model)
            std_list.append(0)

    test_fcm = remove_small_clusters_test(X_val, val_scores, fcm, label_dict, fcm_labels, i)
    fcm_prediction = rnn_local_prediction(std_list, X_val, test_fcm, y_val, i, local_rnn)
    hc_rnn_fcm = r2_score(y_val.sort_index(), fcm_prediction.sort_index())
    print('R2 score for HC-RNN with %.0f clusters: %.2f' % (i, hc_rnn_fcm))

    lda = LDA()
    lda.fit(cal_scores, fcm_labels)
    test_lda = lda.predict(val_scores)

    lda_prediction = rnn_local_prediction(std_list, X_val, test_lda, y_val, i, local_rnn)
    hc_rnn_lda = r2_score(y_val.sort_index(), lda_prediction.sort_index())
    print('R2 score for HC-RNN with %.0f clusters: %.2f' % (i, hc_rnn_lda))

    qda = QDA()
    qda.fit(cal_scores, fcm_labels)
    test_qda = qda.predict(val_scores)

    qda_prediction = rnn_local_prediction(std_list, X_val, test_qda, y_val, i, local_rnn)
    hc_rnn_qda = r2_score(y_val.sort_index(), qda_prediction.sort_index())
    print('R2 score for HC-RNN with %.0f clusters: %.2f' % (i, hc_rnn_qda))

    gnb = GaussianNB()
    gnb.fit(cal_scores, fcm_labels)
    test_gnb = gnb.predict(val_scores)

    gnb_prediction = rnn_local_prediction(std_list, X_val, test_gnb, y_val, i, local_rnn)
    hc_rnn_gnb = r2_score(y_val.sort_index(), gnb_prediction.sort_index())
    print('R2 score for HC-RNN with %.0f clusters: %.2f' % (i, hc_rnn_gnb))

    r2_list = [hc_rnn_fcm, hc_rnn_lda, hc_rnn_qda, hc_rnn_gnb, global_val]
    r2_list = np.array(r2_list)
    
    opt_cluster_rnn.append(r2_list)
    
# =============================================================================
# SVM global model
# =============================================================================
pipe_svr = make_pipeline(SVR())

param_range  = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
param_range2 = ['auto', 'scale']

param_grid   = [{'svr__C': param_range, 'svr__kernel': ['linear']},
                {'svr__C': param_range, 'svr__gamma': param_range2, 'svr__kernel': ['rbf']},
                {'svr__C': param_range, 'svr__gamma': param_range2, 'svr__kernel': ['sigmoid']}]


global_score = []
local_mod = []
local_mse = []
for m in range(len(X_train)):
    global_validation = X_train[X_train.index == index_list[m]]
    global_training = X_train.drop([index_list[m]], axis=0)
    
    global_validation_y = y_train[y_train.index == index_list[m]]
    global_training_y = y_train.drop([index_list[m]], axis=0)   
    
    std_X = StandardScaler(with_mean=True, with_std=True)
    training = std_X.fit_transform(global_training)
    validation = std_X.transform(global_validation)

    pipe_svr = make_pipeline(SVR())
    gs = GridSearchCV(estimator=pipe_svr, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
            
    gs = gs.fit(training, global_training_y[0])
    local_mod.append(gs.best_estimator_)
    local_mse.append(gs.best_score_)   

    score_ = np.stack((local_mse), axis=0)
    score_mean = np.mean(score_, axis=0)
    global_score.append(score_mean)
    
global_score2 = np.array(global_score)    
low_score = global_score2.argmin()
global_params = local_mod[low_score]

std_X = StandardScaler(with_mean=True, with_std=True)
training = std_X.fit_transform(X_train)
validation = std_X.transform(X_test)

svr = global_params
svr_mod = svr.fit(training, y_train[0])
y_pred = svr_mod.predict(validation)
global_svm = r2_score(y_test, y_pred)

# =============================================================================
no_clusters = 2

fcm = FCM(n_clusters=no_clusters, random_state=12)
fcm.fit(X_scores)

fcm_labels = remove_small_clusters(fcm, X_train, no_clusters)[0]
label_dict = remove_small_clusters(fcm, X_train, no_clusters)[1]

print(Counter(fcm_labels))

pipe_svr = make_pipeline(SVR())

param_range  = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
param_range2 = ['auto', 'scale']

param_grid   = [{'svr__C': param_range, 'svr__kernel': ['linear']},
                {'svr__C': param_range, 'svr__gamma': param_range2, 'svr__kernel': ['rbf']},
                {'svr__C': param_range, 'svr__gamma': param_range2, 'svr__kernel': ['sigmoid']}]

# =============================================================================
local_score = []
local_param = []
for k in range(0, no_clusters):
    cluster_k = X_train[fcm_labels == k]
    y_k = y_train[fcm_labels == k]
    
    if len(cluster_k) > 0:
        local_mse = []
        local_mod = []
        interm_pls = []
        index_k = np.unique(cluster_k.index)
        for l in range(len(index_k)):
            validation = cluster_k[cluster_k.index == index_k[l]]
            training = cluster_k.drop([index_k[l]], axis=0)
            
            validation_y = y_k[y_k.index == index_k[l]]
            training_y = y_k.drop([index_k[l]], axis=0)
            
            std_X = StandardScaler(with_mean=True, with_std=True)
            training = std_X.fit_transform(training)
            validation = std_X.transform(validation)
            
            score_comp = []
            pipe_svr = make_pipeline(SVR())
            gs = GridSearchCV(estimator=pipe_svr, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
            
            gs = gs.fit(training, training_y[0])
            local_mod.append(gs.best_estimator_)
            local_mse.append(gs.best_score_)
            
        score_ = np.stack((local_mse), axis=0)
        local_score.append(score_)
        local_param.append(local_mod)
    else:
        print('No samples in cluster %.0f' % k)
        local_score.append([])  
    
local_components = []
for m in range(len(local_score)):
    if len(local_score[m]) > 0:
        ind = local_score[m].argmin()
        params = local_param[m]
        local_components.append(params[ind])
    else:
        local_components.append(0)

local_svm = [] 
std_list = []
for q in range(len(local_components)):
    cluster_q = X_train[fcm_labels == q]
    y_q = y_train[fcm_labels == q]

    svr = local_components[q]
    
    if len(cluster_q) > 0: 
        std_X = StandardScaler(with_mean=True, with_std=True)
        cluster_q = std_X.fit_transform(cluster_q)

        svr_mod = svr.fit(cluster_q, y_q[0])
    
        local_svm.append(svr_mod)
        std_list.append(std_X)
    else:
        local_svm.append(svr)
        std_list.append(0)

# =============================================================================
# Classification and prediction
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_train_std = std_X.fit_transform(X_train)
global_model.fit(X_train_std, y_train)

X_test_std = std_X.transform(X_test)
test_scores = global_model.transform(X_test_std)
X_scores_svm = global_model.transform(X_train_std)

# test_labels = fcm.predict(test_scores)
test_fcm = remove_small_clusters_test(X_test, test_scores, fcm, label_dict, fcm_labels, no_clusters)
fcm_prediction = svm_local_predictions(std_list, test_fcm, X_test, y_test, no_clusters, local_svm)
hc_svm_fcm = r2_score(y_test.sort_index(), fcm_prediction.sort_index())
print('R2 score for HC-SVM with %.0f clusters: %.3f' % (no_clusters, hc_svm_fcm))

# =============================================================================
lda = LDA()
lda.fit(X_scores_svm, fcm_labels)
test_lda = lda.predict(test_scores)

lda_prediction = svm_local_predictions(std_list, test_lda, X_test, y_test, no_clusters, local_svm)
hc_svm_lda = r2_score(y_test.sort_index(), lda_prediction.sort_index())
print('R2 score for HC-SVM with %.0f clusters: %.3f' % (no_clusters, hc_svm_lda))
# =============================================================================
qda = QDA()
qda.fit(X_scores_svm, fcm_labels)
test_qda = qda.predict(test_scores)

qda_prediction = svm_local_predictions(std_list, test_qda, X_test, y_test, no_clusters, local_svm)
hc_svm_qda = r2_score(y_test.sort_index(), qda_prediction.sort_index())
print('R2 score for HC-SVM with %.0f clusters: %.3f' % (no_clusters, hc_svm_qda))
# =============================================================================
gnb = GaussianNB()
gnb.fit(X_scores_svm, fcm_labels)
test_gnb = gnb.predict(test_scores)

gnb_prediction = svm_local_predictions(std_list, test_gnb, X_test, y_test, no_clusters, local_svm)
hc_svm_gnb = r2_score(y_test.sort_index(), gnb_prediction.sort_index())
print('R2 score for HC-SVM with %.0f clusters: %.3f' % (no_clusters, hc_svm_gnb))

# =============================================================================
y_pred = global_model.predict(X_test_std)
global_test_svm = r2_score(y_test, y_pred)
print('R2 score for global PLSR model: %.3f' % global_test_svm) 

# =============================================================================
# Results SVM
# =============================================================================
r2_list_svm = [hc_svm_fcm, hc_svm_lda, hc_svm_qda, hc_svm_gnb, global_test_svm]
r2_list_svm = np.array(r2_list_svm)

# =============================================================================
# Optimal no. clusters SVM
# =============================================================================
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_cal_std = std_X.fit_transform(X_cal)
global_model.fit(X_cal_std, y_cal)
cal_scores = global_model.transform(X_cal_std)

X_val_std = std_X.transform(X_val)
val_scores = global_model.transform(X_val_std)

y_pred = global_model.predict(X_val_std)
global_val = r2_score(y_val, y_pred)
print('R2 calibration-score for global PLSR model: %.3f' % global_val)


pipe_svr = make_pipeline(SVR())

param_range  = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
param_range2 = ['auto', 'scale']

param_grid   = [{'svr__C': param_range, 'svr__kernel': ['linear']},
                {'svr__C': param_range, 'svr__gamma': param_range2, 'svr__kernel': ['rbf']},
                {'svr__C': param_range, 'svr__gamma': param_range2, 'svr__kernel': ['sigmoid']}]

opt_cluster_svm = []
for i in range(2, 11):
    fcm = FCM(n_clusters=i, random_state=12)
    fcm.fit(cal_scores)
    
    fcm_labels = remove_small_clusters(fcm, X_cal, i)[0]
    label_dict = remove_small_clusters(fcm, X_cal, i)[1]
    
    pipe_svr = make_pipeline(SVR())

    param_range  = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    param_range2 = ['auto', 'scale']

    param_grid   = [{'svr__C': param_range, 'svr__kernel': ['linear']},
                    {'svr__C': param_range, 'svr__gamma': param_range2, 'svr__kernel': ['rbf']},
                    {'svr__C': param_range, 'svr__gamma': param_range2, 'svr__kernel': ['sigmoid']}]

    local_score = []
    local_param = []
    for k in range(0, i):
        cluster_k = X_cal[fcm_labels == k]
        y_k = y_cal[fcm_labels == k]
        
        if len(cluster_k) > 0:
            local_mse = []
            local_mod = []
            interm_pls = []
            index_k = np.unique(cluster_k.index)
            for l in range(len(index_k)):
                validation = cluster_k[cluster_k.index == index_k[l]]
                training = cluster_k.drop([index_k[l]], axis=0)
                
                validation_y = y_k[y_k.index == index_k[l]]
                training_y = y_k.drop([index_k[l]], axis=0)
        
                std_X = StandardScaler(with_mean=True, with_std=True)
                training = std_X.fit_transform(training)
                validation = std_X.transform(validation)
                
                score_comp = []
                pipe_svr = make_pipeline(SVR())
                gs = GridSearchCV(estimator=pipe_svr, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
            
                gs = gs.fit(training, training_y[0])
                local_mod.append(gs.best_estimator_)
                local_mse.append(gs.best_score_)
        
            score_ = np.stack((local_mse), axis=0)
            local_score.append(score_)
            local_param.append(local_mod)
        else: 
            print('No samples in cluster %.0f' % k)
            local_score.append([]) 
            local_param.append([])
            
    local_components = []
    for m in range(len(local_score)):
        if len(local_score[m]) > 0:
            ind = local_score[m].argmin()
            params = local_param[m]
            local_components.append(params[ind])
        else:
            local_components.append(0)
    
    local_svm = [] 
    std_list = []
    for q in range(len(local_components)):
        cluster_q = X_cal[fcm_labels == q]
        y_q = y_cal[fcm_labels == q]
        
        svr = local_components[q]
        
        if len(cluster_q) > 0:
            std_X = StandardScaler(with_mean=True, with_std=True)
            cluster_q = std_X.fit_transform(cluster_q)

            svr_mod = svr.fit(cluster_q, y_q[0])
            
            local_svm.append(svr_mod)
            std_list.append(std_X)
        else:
            local_svm.append(svr)
            std_list.append(0)

    test_fcm = remove_small_clusters_test(X_val, val_scores, fcm, label_dict, fcm_labels, i)
    fcm_prediction = svm_local_predictions(std_list, test_fcm, X_val, y_val, i, local_svm)
    hc_svm_fcm = r2_score(y_val.sort_index(), fcm_prediction.sort_index())
    print('R2 score for HC-SVM with %.0f clusters: %.3f' % (i, hc_svm_fcm))

    lda = LDA()
    lda.fit(cal_scores, fcm_labels)
    test_lda = lda.predict(val_scores)
    
    lda_prediction = svm_local_predictions(std_list, test_lda, X_val, y_val, i, local_svm)
    hc_svm_lda = r2_score(y_val.sort_index(), lda_prediction.sort_index())
    print('R2 score for HC-SVM with %.0f clusters: %.3f' % (i, hc_svm_lda))

    qda = QDA()
    qda.fit(cal_scores, fcm_labels)
    test_qda = qda.predict(val_scores)
    
    qda_prediction = svm_local_predictions(std_list, test_qda, X_val, y_val, i, local_svm)
    hc_svm_qda = r2_score(y_val.sort_index(), qda_prediction.sort_index())
    print('R2 score for HC-SVM with %.0f clusters: %.3f' % (i, hc_svm_qda))

    gnb = GaussianNB()
    gnb.fit(cal_scores, fcm_labels)
    test_gnb = gnb.predict(val_scores)

    gnb_prediction = svm_local_predictions(std_list, test_gnb, X_val, y_val, i, local_svm)
    hc_svm_gnb = r2_score(y_val.sort_index(), gnb_prediction.sort_index())
    print('R2 score for HC-SVM with %.0f clusters: %.3f' % (i, hc_svm_gnb))

    r2_list = [hc_svm_fcm, hc_svm_lda, hc_svm_qda, hc_svm_gnb, global_val]
    r2_list = np.array(r2_list)
    
    opt_cluster_svm.append(r2_list)
    
# =============================================================================
# Optimal cluster
# =============================================================================
opt_cluster_all = [opt_cluster_pls, opt_cluster_cnn, opt_cluster_rnn, opt_cluster_svm]

# =============================================================================
# Global models
# =============================================================================
global_models_all = [global_cnn, global_rnn, global_svm]
