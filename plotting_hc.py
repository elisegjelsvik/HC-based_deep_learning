# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:30:48 2022

@author: elgj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.io
from scipy import signal
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from HCDL_functions import remove_small_clusters

df_ftir = scipy.io.loadmat('FTIR_AMW.mat')
df_ftir_prep = scipy.io.loadmat('FTIR_AMW_tidied2.mat')

waves = np.logical_and(df_ftir['waves'] > 700, df_ftir['waves'] < 1800).flatten()
X = df_ftir['spectra']#[:, waves]
y = df_ftir['AMWall']
wave = df_ftir['waves'][waves]

X_clean = df_ftir['spectra'][:, waves]

rep = df_ftir_prep['replicates']
rep_nr = pd.DataFrame(rep)[0]
sample_class = pd.Series(df_ftir_prep['classes'][:,0])
sample_class.index = rep_nr
print(Counter(sample_class))

df = pd.DataFrame(X)
df.index = rep_nr

df_clean = pd.DataFrame(X_clean)
df_clean.index = rep_nr

y_target = pd.Series(y[:, 0])
y_target.index = rep_nr

class_groups = df_ftir_prep['classNames'] 

animal_group = pd.Series(df_ftir_prep['animal'][:,0])
animal_group.index = rep_nr

animal_type = pd.Series(df_ftir_prep['Names1'])
animal_type.index = rep_nr

df_ftir['material'] -= 1
material = df_ftir['material']
enzyme_name = df_ftir['materName']

material2 = pd.DataFrame(material)
material2.index = rep_nr

time_inf = pd.DataFrame(df_ftir['t3'])
time_inf.index = rep_nr

# =============================================================================
# Average over replicas
# =============================================================================
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

df_avg = average_calc(df)
y_target = average_calc(pd.DataFrame(y_target))
sample_class = average_calc(pd.DataFrame(material2)).astype(int)
animal_group = average_calc(pd.DataFrame(animal_group))
time_inf = average_calc(time_inf).astype(int)

df_clean = average_calc(df_clean)

# df3 = average_calc(df3)
# y_target2 = average_calc(pd.DataFrame(y_target2))
# sample_class2 = average_calc(pd.DataFrame(sample_class2))

X_train, X_test, y_train, y_test = train_test_split(df_avg, y_target, test_size=0.5, random_state=1, stratify=sample_class)
train_index = X_train.index
test_index = X_test.index

X_train_class = sample_class.loc[X_train.index]
X_train_animal = animal_group.loc[X_train.index]
X_train_type = animal_type.loc[X_train.index]
X_train_time = time_inf.loc[X_train.index]

# =============================================================================
# Std.Error over replicas
# =============================================================================
# sqrt(mean(var(betweenReplicas)/nReps))

index_list = np.unique(X_train.index)

std_error = []
for s in range(len(index_list)):
    rep = df[df.index == index_list[s]]
    
    if len(rep) > 1:
        rep_error = np.sqrt(np.mean(np.var(rep)/len(rep)))
        std_error.append(rep_error)
    else:
        std_error.append(0)
    
plt.bar(index_list, std_error)

# =============================================================================
# Enzyme
# =============================================================================
Alcalase = ['CMDRA  ', 'hCMDRA ', 'CMA    ', 'CSA    ', 'CBA    ', 'TCA    ', 'MTDRA  ',
            'SHA    ', 'SSA    ', 'SBA    ', 'MaA    ']
Papain = ['CMDRPa ', 'hCMDRPa', 'CMPa   ', 'CSPa   ', 'CBPa   ', 'MaPa   ']
Protamex = ['CMDRPr ', 'hCMDRPr', 'CMPr   ', 'CSPr   ', 'CBPr   ']
Corolase = ['TCC    ', 'MTDRC  ']
Flavorzyme = ['TCF    ', 'MTDRF  ', 'MaF    ']
Ingen = ['Ma     ']

def get_enzyme(material):
    species = []
    enzyme_class = []
    for sample in material:
        if sample in Alcalase:
            species.append("Alcalase")
            enzyme_class.append(1)
        elif sample in Papain:
            species.append("Papain")
            enzyme_class.append(2)
        elif sample in Protamex:
            species.append("Protamex")
            enzyme_class.append(3)
        elif sample in Corolase:
            species.append("Corolase")
            enzyme_class.append(4)
        elif sample in Flavorzyme:
            species.append("Flavorzyme")
            enzyme_class.append(5)
        elif sample in Ingen:
            species.append("NaN")
            enzyme_class.append(6)

    if len(material) == len(species):
        return species, enzyme_class
    else:
        return "Error list has the wrong length"

X_train_enz = []
for i in X_train_class[0]:
    X_train_enz.append(enzyme_name[i])
    
X_train_enz = np.array(X_train_enz)
np.unique(X_train_enz)

X_train_enzyme_list = get_enzyme(X_train_enz)[0]
X_train_enzyme_class = get_enzyme(X_train_enz)[1] 

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
import numpy.linalg as la
from scipy.special import eval_legendre

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

X_train = np.array(X_train)
X_train = signal.savgol_filter(X_train, window_length=11, polyorder=3, deriv=2)

ref_spec = np.mean(X_train, axis=0) 
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
# Plots for Harald
# =============================================================================
m_turkey = np.mean(X_train[X_train_animal.iloc[:, 0] == 2])
m_chick = np.mean(X_train[X_train_animal.iloc[:, 0] == 1])
m_salmon = np.mean(X_train[X_train_animal.iloc[:, 0] == 4])
m_mack = np.mean(X_train[X_train_animal.iloc[:, 0] == 3])

m_animals = pd.concat([m_turkey, m_chick, m_salmon, m_mack], axis=1).T
m_animals.index = ['Turkey', 'Chicken', 'Salmon', 'Mackerel']

X_train_clean = df_clean.loc[X_train.index]
clean_turkey = np.mean(X_train_clean[X_train_animal.iloc[:, 0]==2])
clean_chick = np.mean(X_train_clean[X_train_animal.iloc[:, 0]==1])
clean_salmon = np.mean(X_train_clean[X_train_animal.iloc[:, 0]==4])
clean_mack = np.mean(X_train_clean[X_train_animal.iloc[:, 0]==3])

clean_animals = pd.concat([clean_turkey, clean_chick, clean_salmon, clean_mack], axis=1).T
clean_animals.index = ['Turkey', 'Chicken', 'Salmon', 'Mackerel']

X_tenz = pd.DataFrame(X_train_enzyme_class)
X_tenz.index = train_index
e_1 = np.mean(X_train[X_tenz.iloc[:,0] == 1])
e_2 = np.mean(X_train[X_tenz.iloc[:,0] == 2])
e_3 = np.mean(X_train[X_tenz.iloc[:,0] == 3])
e_4 = np.mean(X_train[X_tenz.iloc[:,0] == 4])
e_5 = np.mean(X_train[X_tenz.iloc[:,0] == 5])
e_6 = np.mean(X_train[X_tenz.iloc[:,0] == 6])

enzyme_animals = pd.concat([e_1, e_2, e_3, e_4, e_5, e_6], axis=1).T
enzyme_animals.index = ['Alcalase', 'Papain', 'Protamex', 'Corolase 2TS', 'Flavorzyme', 'Nan']

c_1 = np.mean(X_train_clean[X_tenz.iloc[:,0] == 1])
c_2 = np.mean(X_train_clean[X_tenz.iloc[:,0] == 2])
c_3 = np.mean(X_train_clean[X_tenz.iloc[:,0] == 3])
c_4 = np.mean(X_train_clean[X_tenz.iloc[:,0] == 4])
c_5 = np.mean(X_train_clean[X_tenz.iloc[:,0] == 5])
c_6 = np.mean(X_train_clean[X_tenz.iloc[:,0] == 6])

enzyme_clean = pd.concat([c_1, c_2, c_3, c_4, c_5, c_6], axis=1).T
enzyme_clean.index = ['Alcalase', 'Papain', 'Protamex', 'Corolase', 'Flavorzyme', 'Nan']

fig, ax = plt.subplots(2,2, figsize=(6,6), dpi=300)
ax[0,0].plot(wave, clean_turkey.T, label="Turkey")
ax[0,0].plot(wave, clean_chick.T, label="Chicken")
ax[0,0].plot(wave, clean_salmon.T, label="Salmon")
ax[0,0].plot(wave, clean_mack.T, label="Mackerel")

ax[0,1].plot(wave, m_turkey.T, label="Turkey")
ax[0,1].plot(wave, m_chick.T, label="Chicken")
ax[0,1].plot(wave, m_salmon.T, label="Salmon")
ax[0,1].plot(wave, m_mack.T, label="Mackerel")

ax[1,0].plot(wave, c_1.T, label="Alcalase")
ax[1,0].plot(wave, c_2.T, label="Papain")
ax[1,0].plot(wave, c_3.T, label="Protamex")
ax[1,0].plot(wave, c_4.T, label="Coralse 2TS")
ax[1,0].plot(wave, c_5.T, label="Flavorzyme")
ax[1,0].plot(wave, c_6.T, label="Mackerel")

ax[1,1].plot(wave, e_1.T, label="Alcalase")
ax[1,1].plot(wave, e_2.T, label="Papain")
ax[1,1].plot(wave, e_3.T, label="Protamex")
ax[1,1].plot(wave, e_4.T, label="Coralse 2TS")
ax[1,1].plot(wave, e_5.T, label="Flavorzyme")
ax[1,1].plot(wave, e_6.T, label="Mackerel")

ax[0,0].set_title('Raw spectra')
ax[0,1].set_title('Preprocessed spectra')
ax[1,0].set_title('Raw spectra')
ax[1,1].set_title('Preprocessed spectra')

ax[0,0].set_title('A)', loc='left', fontsize='medium')
ax[0,1].set_title('B)', loc='left', fontsize='medium')
ax[1,0].set_title('C)', loc='left', fontsize='medium')
ax[1,1].set_title('D)', loc='left', fontsize='medium')

ax[0,0].legend(fancybox=True, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, prop={'size': 7})
ax[0,1].legend(fancybox=True, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, prop={'size': 7})
ax[1,0].legend(fancybox=True, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, prop={'size': 7})
ax[1,1].legend(fancybox=True, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, prop={'size': 7})

ax[0,0].set_xlabel("Wavenumber ($cm^{-1}$)")
ax[0,0].set_ylabel("Absorbance")
ax[0,1].set_xlabel("Wavenumber ($cm^{-1}$)")
ax[0,1].set_ylabel("Absorbance")
ax[1,0].set_xlabel("Wavenumber ($cm^{-1}$)")
ax[1,0].set_ylabel("Absorbance")
ax[1,1].set_xlabel("Wavenumber ($cm^{-1}$)")
ax[1,1].set_ylabel("Absorbance")

ax[0,0].invert_xaxis()
ax[0,1].invert_xaxis()
ax[1,0].invert_xaxis()
ax[1,1].invert_xaxis()

plt.tight_layout()
plt.show()

# =============================================================================
mean_spec = ref_spec[waves]

fig, ax = plt.subplots(2,2, figsize=(6,6), dpi=300)
ax[0,0].scatter(m_turkey, mean_spec)
ax[0,1].scatter(m_chick, mean_spec)
ax[1,0].scatter(m_salmon, mean_spec)
ax[1,1].scatter(m_mack, mean_spec)

ax[0,0].set_title("Turkey")
ax[0,1].set_title("Chicken")
ax[1,0].set_title("Salmon")
ax[1,1].set_title("Mackerel")

plt.tight_layout()
plt.show()

clean_mean = np.mean(X_train_clean)

fig, ax = plt.subplots(2,2, figsize=(6,6), dpi=300)
ax[0,0].scatter(clean_turkey, clean_mean)
ax[0,1].scatter(clean_chick, clean_mean)
ax[1,0].scatter(clean_salmon, clean_mean)
ax[1,1].scatter(clean_mack, clean_mean)

ax[0,0].set_title("Turkey")
ax[0,1].set_title("Chicken")
ax[1,0].set_title("Salmon")
ax[1,1].set_title("Mackerel")

plt.tight_layout()
plt.show()

# =============================================================================
# Correct HC-PLSR
# =============================================================================
index_list = np.unique(X_train.index)            

# global_comp = []
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
    
expl_var = explained_variance_pls(pls, X_train_std) #Hva menes med kryssvalidert forklart varians?

fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
plt.bar(range(1, len(expl_var)+1), expl_var*100)
plt.title('Screeplot')
plt.ylabel('Explained variance (%)')
plt.xlabel('No. components')
plt.tight_layout()
plt.show()

# =============================================================================
# Scoreplots from global model
# =============================================================================
global_model = PLSRegression(n_components=global_n, scale=False)
std_X = StandardScaler(with_mean=True, with_std=False)
X_train_std = std_X.fit_transform(X_train)
global_model.fit(X_train_std, y_train)
X_scores = pls.transform(X_train_std)

animal_label = ['Chicken', 'Turkey', 'Mackerel', 'Salmon']
enzyme_list = ['Alcalase', 'Papain', 'Protamex', 'Corolase 2TS', 'Flavorzyme', 'Nan']
time_list = ['5', '25', '50', '75', '100', '150', '200', '300', '400', '500', '600', '800']

fig, ax = plt.subplots(2, 2, figsize=(6,5.8), dpi=300) #3.5 i høyde for enkelt plot
scatter0 = ax[0,0].scatter(X_scores[:,0], X_scores[:,1], c=X_train_enzyme_class, alpha=.7, cmap='rainbow')
scatter1 = ax[0,1].scatter(X_scores[:,0], X_scores[:,1], c=X_train_animal, alpha=.7, cmap='rainbow')
ax[1,0].scatter(X_scores[:,0], X_scores[:,2], c=X_train_enzyme_class, alpha=.7, cmap='rainbow')
ax[1,1].scatter(X_scores[:,0], X_scores[:,2], c=X_train_animal, alpha=.7, cmap='rainbow')
ax[0,0].set_title('Enzymes')
ax[1,0].set_title('Enzymes')
ax[0,0].set_title('A)', loc='left', fontsize='medium')
ax[0,1].set_title('Animal')
ax[1,1].set_title('Animal')
ax[0,1].set_title('B)', loc='left', fontsize='medium')
ax[1,0].set_title('C)', loc='left', fontsize='medium')
ax[1,1].set_title('D)', loc='left', fontsize='medium')
ax[0,0].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[0,0].set_ylabel('PC2 ({} %)'.format(round(expl_var[1]*100, 2)))
ax[0,1].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[0,1].set_ylabel('PC2 ({} %)'.format(round(expl_var[1]*100, 2)))
ax[1,0].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[1,0].set_ylabel('PC3 ({} %)'.format(round(expl_var[2]*100, 2)))
ax[1,1].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[1,1].set_ylabel('PC3 ({} %)'.format(round(expl_var[2]*100, 2)))
ax[0,0].legend(handles=scatter0.legend_elements()[0], labels=enzyme_list, fancybox=True,
             loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, prop={'size': 7})
ax[0,1].legend(handles=scatter1.legend_elements()[0], labels=animal_label, fancybox=True,
             loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, prop={'size': 7})
plt.suptitle('PLSR of training data')
plt.tight_layout()
plt.show()

# =============================================================================
fig, ax = plt.subplots(3, 2, figsize=(6,7), dpi=300) 
scatter0 = ax[0,0].scatter(X_scores[:,0], X_scores[:,1], c=X_train_enzyme_class, alpha=.7, cmap='rainbow')
scatter1 = ax[0,1].scatter(X_scores[:,0], X_scores[:,1], c=X_train_animal, alpha=.7, cmap='rainbow')
scatter2 = ax[2,0].scatter(X_scores[:,0], X_scores[:,1], c=X_train_time, alpha=.7, cmap='rainbow')
ax[1,0].scatter(X_scores[:,0], X_scores[:,2], c=X_train_enzyme_class, alpha=.7, cmap='rainbow')
ax[1,1].scatter(X_scores[:,0], X_scores[:,2], c=X_train_animal, alpha=.7, cmap='rainbow')
ax[2,1].scatter(X_scores[:,0], X_scores[:,2], c=X_train_time, alpha=.7, cmap='rainbow')
ax[0,0].set_title('Enzymes')
ax[0,0].set_title('A)', loc='left', fontsize='medium')
ax[0,1].set_title('Animal')
ax[0,1].set_title('B)', loc='left', fontsize='medium')
ax[1,0].set_title('C)', loc='left', fontsize='medium')
ax[1,1].set_title('D)', loc='left', fontsize='medium')
ax[0,0].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[0,0].set_ylabel('PC2 ({} %)'.format(round(expl_var[1]*100, 2)))
ax[0,1].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[0,1].set_ylabel('PC2 ({} %)'.format(round(expl_var[1]*100, 2)))
ax[1,0].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[1,0].set_ylabel('PC3 ({} %)'.format(round(expl_var[2]*100, 2)))
ax[1,1].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[1,1].set_ylabel('PC3 ({} %)'.format(round(expl_var[2]*100, 2)))
ax[2,0].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[2,0].set_ylabel('PC2 ({} %)'.format(round(expl_var[1]*100, 2)))
ax[2,1].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[2,1].set_ylabel('PC3 ({} %)'.format(round(expl_var[2]*100, 2)))
ax[0,0].legend(handles=scatter0.legend_elements()[0], labels=enzyme_list, fancybox=True,
             loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, prop={'size': 7})
ax[0,1].legend(handles=scatter1.legend_elements()[0], labels=animal_label, fancybox=True,
             loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, prop={'size': 7})
ax[2,0].legend(handles=scatter2.legend_elements()[0], labels=time_list, fancybox=True,
             loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, prop={'size': 7})
plt.suptitle('PLSR of training data')
plt.tight_layout()
plt.show()


# =============================================================================

fig, ax = plt.subplots(2, 1, figsize=(6,7), dpi=300) 
scatter2 = ax[0].scatter(X_scores[:,0], X_scores[:,1], c=X_train_time, alpha=.7, cmap='rainbow')
ax[0].scatter(X_scores[:,0], X_scores[:,2], c=X_train_time, alpha=.7, cmap='rainbow')
ax[0].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[0].set_ylabel('PC2 ({} %)'.format(round(expl_var[1]*100, 2)))
ax[1].set_xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
ax[1].set_ylabel('PC3 ({} %)'.format(round(expl_var[2]*100, 2)))

ax[0].legend(handles=scatter2.legend_elements()[0], labels=time_list,
             loc='upper center', bbox_to_anchor=(1, 0.95), ncol=4, prop={'size': 7})
plt.suptitle('PLSR of training data')
plt.tight_layout()
plt.show()

# =============================================================================

count=1
fig = plt.figure(figsize=(6, 20), dpi=300)
for i in range(2,11):
    fcm = FCM(n_clusters=i, random_state=12)
    fcm.fit(X_scores)
    
    fcm_centers = fcm.centers
    fcm_labels  = remove_small_clusters(fcm, X_train, i)[0]
    
    plt.subplot(10,2,count)
    plt.scatter(X_scores[:,0], X_scores[:,1], c=fcm_labels, alpha=.8, cmap='rainbow')
    # plt.scatter(fcm_centers[:,0], fcm_centers[:,1], cmap='rainbow', marker="s", s=50, c='black')
    plt.title('Number of clusters: %.0f' % i)
    plt.xlabel('PC1 ({} %)'.format(round(expl_var[0]*100, 2)))
    plt.ylabel('PC2 ({} %)'.format(round(expl_var[1]*100, 2)))
    count+=1

plt.suptitle('Fuzzy C-Means Clustering', y=0.99)
plt.tight_layout()
plt.show()

# =============================================================================
# Plottong optimal number of clusters
# =============================================================================
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

label_list = ['A)', 'B)', 'C)', 'D)']

opt_cluster_dict = pickle.load(open('optimal_cluster.p', 'rb'))
opt_cluster_pls = pd.DataFrame(opt_cluster_dict[0])
opt_cluster_cnn = pd.DataFrame(opt_cluster_dict[1])
opt_cluster_rnn = pd.DataFrame(opt_cluster_dict[2])
opt_cluster_svm = pd.DataFrame(opt_cluster_dict[3])

fig, ax = plt.subplots(4,1, figsize=(5,8.7), dpi=300)
ax[0].plot(opt_cluster_pls.iloc[:,0:4])
ax[1].plot(opt_cluster_cnn.iloc[:,0:4])
ax[2].plot(opt_cluster_rnn.iloc[:,0:4])
ax[3].plot(opt_cluster_svm.iloc[:,0:4])
plt.suptitle('Optimal cluster')
for f in range(len(ax)):
    ax[f].set_xlabel('No. clusters')
    ax[f].set_ylabel('R-squared')
    ax[f].set_xticks(np.arange(9))
    ax[f].set_xticklabels(list(range(2,11)))
    ax[f].legend(['FCM', 'LDA', 'QDA', 'GBN'], loc='lower left')
    ax[f].set_title(label_list[f], loc='left', fontsize='medium')
    # ax[f].set_ylim(-0.55, 0.9)
ax[0].set_title('HC-PLSR')
ax[1].set_title('HC-CNN')
ax[2].set_title('HC-RNN')
ax[3].set_title('HC-SVR')
plt.tight_layout()
plt.show()

# fig.savefig('File_title.eps', format='eps', bbox_inches='tight')

# =============================================================================
from scipy import stats

def optimal_cluster(cluster_list):
    best_cluster = []
    for c in range(len(cluster_list.T)):
        best_cluster.append(cluster_list[c].argmax()+2)
    return best_cluster

# Finds optimal cluster based on mode (typetall)
opt_pls = stats.mode(optimal_cluster(opt_cluster_pls), keepdims=False)[0]
opt_cnn = stats.mode(optimal_cluster(opt_cluster_cnn), keepdims=False)[0]
opt_rnn = stats.mode(optimal_cluster(opt_cluster_rnn), keepdims=False)[0]
opt_svm = stats.mode(optimal_cluster(opt_cluster_svm), keepdims=False)[0]

# Finds optimal cluster based on mean of each cluster
opt_pls = np.mean(opt_cluster_pls.T).argmax()+2
opt_cnn = np.mean(opt_cluster_cnn.T).argmax()+2
opt_rnn = np.mean(opt_cluster_rnn.T).argmax()+2
opt_svm = np.mean(opt_cluster_svm.T).argmax()+2

# =============================================================================
# Plotting R-squared results 
# =============================================================================
r2_pls = pickle.load(open('r2_pls.p', 'rb'))
r2_cnn = pickle.load(open('r2_cnn.p', 'rb'))
r2_rnn = pickle.load(open('r2_rnn.p', 'rb'))
r2_svm = pickle.load(open('r2_svm.p', 'rb'))

r2_all = []
r2_all.append(r2_pls[0:4])
r2_all.append(r2_cnn[0:4])
r2_all.append(r2_rnn[0:4])
r2_all.append(r2_svm[0:4])

r2_all = pd.DataFrame(r2_all)

barWidth = 0.10

bars1 = r2_all[0]
bars2 = r2_all[1]
bars3 = r2_all[2]
bars4 = r2_all[3]

global_models = pickle.load(open('global_r2_scores.p', 'rb'))
global_pls = global_models[0]
global_cnn = global_models[1]
global_rnn = global_models[2]
global_svm = global_models[3]

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r6 = 4
r7 = 4.5
r8 = 5
r9 = 5.5
global_ticks = [r6, r7, r8, r9]

x_tick = [r + barWidth*2 for r in range(len(bars1))]
x_tick = x_tick + global_ticks

fig, ax = plt.subplots(figsize=(12,5), dpi=300)
plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='FCM')
plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='LDA')
plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='QDA')
plt.bar(r4, bars4, width=barWidth, edgecolor='white', label='GNB')
plt.bar(r6, global_pls, width=barWidth, edgecolor='white')
plt.bar(r7, global_cnn, width=barWidth, edgecolor='white')
plt.bar(r8, global_rnn, width=barWidth, edgecolor='white')
plt.bar(r9, global_svm, width=barWidth, edgecolor='white')

plt.xlabel('HC-method')
plt.ylabel('R-squared')
plt.xticks(x_tick, ['HC-PLSR', 'HC-CNN', 'HC-RNN', 'HC-SVM', 'Global PLSR', 'Global CNN', 'Global RNN', 'Global SVM'])
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.7))
plt.tight_layout()
plt.title('R-squared for HC-methods')
plt.show()

# =============================================================================
# Feature visualisation from CNN
# =============================================================================
import seaborn as sns
import string

cnn_vis = pickle.load(open('cnn_feature_vis.p', 'rb'))
rnn_vis = pickle.load(open('rnn_feature_vis.p', 'rb'))

sns.heatmap(pd.DataFrame(cnn_vis[0]).T, cmap='viridis')
plt.show()

def cluster_samples(X_train, fcm_labels):
    cluster_list = []
    for s in range(len(np.unique(fcm_labels))):
        cluster_list.append(np.mean(X_train[fcm_labels == s]))
    return cluster_list

cluster_list = cluster_samples(X_train, fcm_labels)

def heatmap_vis(gradients, X_inn, title):
    label_list = list(string.ascii_uppercase)    
    count = 1
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    for i in range(len(gradients)):
        plt.subplot(2,1, count)
        plt.tick_params(bottom='on')
        
        grad_res = pd.DataFrame(gradients[i]).T
        grad_res.columns = wave[:,0]
        
        plt.xlabel('Wavenumber ($cm^{-1}$)')
        plt.margins(x=0)
        plt.title('Cluster %.0f' % (i + 1))
        plt.title('%s)' % label_list[i], loc='left', fontsize='medium')
        
        im = plt.imshow(grad_res, cmap='viridis', aspect="auto", extent=[1799,700,-0.003,0.003])
        
        ax2 = plt.twinx()
        plt.plot(wave, X_inn[i], linewidth=2, color='black')
        
        ax.axis('tight') 
        plt.colorbar(im, pad=0.11)
        
        count+=1
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig

cnn_fig = heatmap_vis(cnn_vis, cluster_list, "CNN feature visualisation")
rnn_fig = heatmap_vis(rnn_vis, cluster_list, "RNN feature visualisation")

# =============================================================================
# PLSR loadings
# =============================================================================
def vis_plsr_loadings(local_pls, title):
    label_list = list(string.ascii_uppercase)  
    count = 1
    fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
    for i in range(len(local_pls)):
        plt.subplot(len(local_pls), 1, count)
        plt.tick_params(bottom='on')
        
        plt.margins(x=0)
        plt.xlabel('Wavenumber ($cm^{-1}$)')
        model = local_pls[i]
        pls_loadings = model.x_loadings_[:,0]
        pls_res = pd.DataFrame(pls_loadings).T
        pls_res.columns = wave[:,0]
        
        im = plt.imshow(pls_res, cmap='viridis', aspect="auto", extent=[1799,700,-0.003,0.003])
        
        ax2 = plt.twinx()    
        plt.plot(wave, cluster_list[i], color='black')
        plt.title('Cluster %.0f' % (i + 1))
        plt.title('%s)' % label_list[i], loc='left', fontsize='small')
        
        ax.axis('tight') 
        plt.colorbar(im, pad=0.11)
        
        count += 1
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig

pls_fig = vis_plsr_loadings(local_pls, 'PLSR Loadings')

# =============================================================================
# Bruke permutation importance for SVR
# =============================================================================
from sklearn.inspection import permutation_importance

y_cluster0 = y_train[fcm_labels == 0]
y_cluster1 = y_train[fcm_labels == 1]
y_cluster_list = [y_cluster0, y_cluster1]

X_clust0 = X_train[fcm_labels == 0]
X_clust1 = X_train[fcm_labels == 1]
X_cluster_list = [X_clust0, X_clust1]

def vis_svr_permutations(local_svm, title):
    count = 1
    label_list = list(string.ascii_uppercase)  
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    for i in range(len(local_svm)):
        plt.subplot(len(local_svm), 1, count)
        plt.tick_params(bottom='on')
        
        plt.margins(x=0)
        plt.xlabel('Wavenumber ($cm^{-1}$)')
        
        r = permutation_importance(local_svm[i], X_cluster_list[i], y_cluster_list[i], n_repeats=20, random_state=1)
        
        pi = pd.DataFrame(r['importances_mean']).T
        pi.columns = wave[:,0]
        
        im = plt.imshow(pi, cmap='viridis', aspect="auto", extent=[1799,700,-0.003,0.003])
        
        ax2 = plt.twinx()  
        plt.plot(wave, cluster_list[i], color='black')
        plt.title('Cluster %.0f' % (i + 1))
        plt.title('%s)' % label_list[i], loc='left', fontsize='medium')
        
        ax.axis('tight') 
        plt.colorbar(im, pad=0.11)
        
        count += 1    

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig

svr_fig = vis_svr_permutations(local_svm, 'SVR feature permutation importance')