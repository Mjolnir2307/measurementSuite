######## Importing libraries
import os                                                                                                         
import gc
import math
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import normalize as norm

###### Delta-Distance Score
def delta_dist_compute(embeddings,g_id,num_subjects,y_dev,y_dev_id):

    """
    Function to Compute Delta Distance for a Gesture

    INPUTS:-
    1) embeddings: Feature Embeddings
    2) g_id: Index Value of the Gesture Class
    3) num_subjects: Number of Subjects
    4) y_dev: Ground-Truth Gesture Labels
    5) y_dev_id: Ground-Truth Subject Labels

    OUTPUTS:-
    1) delta_dist: |d_cs - d_c|/d_c, Delta Distance Computed for a particular gesture

    """
    ###### Defining Essentials
    d_cs = [] # List to Maximal Intra-Subject Distance

    ###### Iterating over Subjects
    for s_id in range(num_subjects): 

        ##### Defining Essentials
        embedding_store_s = [] # List to Store Embeddings from Gesture - 'g_id' and Subject - 's_id'
        dist_store_s = [] # List to Store Distance within subject 's_id' 

        ##### Curating Required Gesture List from g_id and s_id
        for idx in range(y_dev.shape[0]): # Iterating over Embeddings

            if(y_dev[idx] == g_id and y_dev_id[idx] == s_id):
                embedding_store_s.append(embeddings[idx]) # Storing the Required Embeddings
        
        ##### Computing Intra-Gesture and Intra-Subject Distances
        for emb_query_idx, emb_query in enumerate(embedding_store_s):

            if(emb_query_idx != (len(embedding_store_s)-1)): # Checking if the Current Query is the Last Query

                for emb_key_idx in range(emb_query_idx+1,len(embedding_store_s),1): # Iterating over the Embeddings

                    emb_key_curr = embedding_store_s[emb_key_idx] # Extracting Current Embedding Key
                    dist_curr = distance.euclidean(emb_query,emb_key_curr) # Current Distance 
                    dist_store_s.append(dist_curr) # Appending the Computed Distance to dist_curr

        ##### Computing Maximal Distance for the Current Gesture and Subject
        d_cs_curr = np.max(dist_store_s)
        d_cs.append(d_cs_curr) # Storing Values

    ###### Computing Delta Distance
    d_c = np.max(d_cs) # Maximal Distance Amongst all the Subjects
    print('Average Intra-Subject Separation: '+str(np.average(d_cs)))
    print('Minimum Intra-Subject Separation: '+str(np.min(d_cs)))
    print('Maximum Intra-Subject Separation (d_cs): '+str(d_c))
    delta_dist = np.average(np.abs(d_cs - d_c)/d_c) # Averaging Computed Delta Distance Per Subject
    
    return delta_dist