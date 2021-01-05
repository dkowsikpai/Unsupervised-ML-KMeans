# version 1.1

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
    
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score,davies_bouldin_score

import warnings
warnings.filterwarnings('ignore')
    
def cluster_range(X, clusterer, k_start, k_stop, actual=None):
    """
    Generates a dictionary of cluster labels, internal validation values,
    and external validation values for every k

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point

    clusterer : clustering object
        The clustering method used

    k_start : integer
        Initial value to step through

    k_stop : integer
        Final value to step through

    actual : list, optional
        List of labels

    Returns
    ----------
    dictionary
        Contains cluster labels, internal and external validation values for every k
    """
    # empty arrays for the 4 validation criteria
    chs, dbi, inertias, scs = [], [], [], []

    for k in range(k_start, k_stop+1):
        clusterer.n_clusters = k
        X_predict = clusterer.fit_predict(X)

        chs.append(calinski_harabaz_score(X, X_predict))   # Calinski-Harabasz Index
        dbi.append(davies_bouldin_score(X, X_predict))     # Davies-Bouldin Index
        inertias.append(clusterer.inertia_)                # Inertia or within-cluster sum-of-squares criterion
        scs.append(silhouette_score(X, X_predict))         # Silhouette Coefficient

    res = {'chs': chs,
           'dbi': dbi,
           'inertias': inertias,
           'scs': scs}
    
    return res

def plot_internal(inertias, chs, dbi, scs):
    """Plot internal validation values"""
    
    fig, ax = plt.subplots(2,2, figsize=(15,10))
    ks = np.arange(2, len(inertias)+2)
    
    ax[0,0].plot(ks, inertias, '-o', label='Inertia')
    ax[0,1].plot(ks, chs, '-ro', label='Calinski-Harabasz Index')
    ax[1,0].plot(ks, dbi, '-go', label='Davies-Bouldin Index')
    ax[1,1].plot(ks, scs, '-ko', label='Silhouette coefficient')
    
    ax[0,0].set_xlabel('$k$')
    ax[0,0].set_ylabel('Inertia')
    
    ax[0,1].set_xlabel('$k$')
    ax[0,1].set_ylabel('Calinski-Harabasz Index')
    
    ax[1,0].set_ylabel('Davies-Bouldin Index')
    ax[1,0].set_xlabel('$k$')
    
    ax[1,1].set_ylabel('Silhouette')
    ax[1,1].set_xlabel('$k$')
    
    return ax
