import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st

TARGET_RSI_VALUES = [30, 45, 55, 75]
NUM_CLUSTERS = 4
INITIAL_CENTROIDS = np.zeros((len(TARGET_RSI_VALUES), 18))

# Set initial centroids based on target RSI values
INITIAL_CENTROIDS[:, 1] = TARGET_RSI_VALUES


def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=NUM_CLUSTERS,
                           random_state=0,
                           init=INITIAL_CENTROIDS).fit(df).labels_
    return df

def run_k_means_algorithm(df):
    return df.dropna().groupby('date', group_keys=False).apply(get_clusters)

def plot_clusters(data):
    plt.clf()

    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,5] , cluster_0.iloc[:,1] , color = 'red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,5] , cluster_1.iloc[:,1] , color = 'green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,5] , cluster_2.iloc[:,1] , color = 'blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,5] , cluster_3.iloc[:,1] , color = 'black', label='cluster 3')
    
    plt.legend()
    st.pyplot(plt)


