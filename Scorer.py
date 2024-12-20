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
from DGBQA_Score import gbqa_delta_dist_compute
from Delta_Distance import delta_dist_compute
from Generative_Capacity import get_cosine_bounds, ratio_hyperspherical_caps
from MasterFace_Capacity import Compute_MasterFace_Capacity
from ICGDScore import CGID_Score_Calculator
from RankDeviation import avg_rank_deviation
from AcceptanceScore import acceptance_score
from PatternMatchDistance import pattern_match_dist
from AcceptanceScoreComparison import acceptance_score_comp

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name",
                    type=str,
                    help="Name of the Experiment being run, will be used saving the model and correponding outputs")

args = parser.parse_args()

####### Score estimation

##### Defining Essentials
gesture_list = ['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold']
num_subjects = 10
num_gestures = 11
dgbqa_score = []
d_c_star = []
d_cs = []
dgbqa_score_wo = []
Test_Embeddings = np.load('./Embeddings/'+str(args.exp_name)+'.npz')['arr_0']
y_dev = np.load('./Embeddings/'+str(args.exp_name)+'.npz')['arr_0']
y_dev_id = [] 

##### DGBQA Score
for g_id, gesture_curr in enumerate(gesture_list):
    print('==============================================')
    dgbqa_score_curr, d_c_star_curr, d_cs_curr, dgbqa_score_wo_curr = gbqa_delta_dist_compute(Test_Embeddings,g_id,num_subjects,y_dev,y_dev_id)
    dgbqa_score.append(dgbqa_score_curr)
    d_c_star.append(d_c_star_curr)
    d_cs.append(d_cs_curr)
    dgbqa_score_wo.append(dgbqa_score_wo_curr)
    print('GBQA Delta Distance for '+str(gesture_curr)+' = '+str(dgbqa_score_curr))  

dgbqa_score = np.array(dgbqa_score) # Array Formation
dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

d_c_star = np.array(d_c_star) # Array Formation
d_c_star = (d_c_star - np.mean(d_c_star))/np.std(d_c_star) # Mean Normalization
d_c_star = d_c_star/np.linalg.norm(d_c_star) # L2-Normalization

d_cs = np.array(d_cs) # Array Formation
d_cs = (d_cs - np.mean(d_cs))/np.std(d_cs) # Mean Normalization
d_cs = d_cs/np.linalg.norm(d_cs) # L2-Normalization

dgbqa_score_wo = np.array(dgbqa_score_wo) # Array Formation
dgbqa_score_wo = (dgbqa_score_wo - np.mean(dgbqa_score_wo))/np.std(dgbqa_score_wo) # Mean Normalization
dgbqa_score_wo = dgbqa_score_wo/np.linalg.norm(dgbqa_score_wo) # L2-Normalization

##### Delta Distance
delta_distance = []
for g_id, gesture_curr in enumerate(gesture_list):
    print('==============================================')
    delta_dist_g = delta_dist_compute(Test_Embeddings,g_id,num_subjects,y_dev,y_dev_id)
    delta_distance.append(delta_dist_g)
    print('Delta Distance for '+str(gesture_curr)+' = '+str(delta_dist_g))

delta_distance = np.array(delta_distance) # Array Formation
delta_distance = (delta_distance - np.mean(delta_distance))/np.std(delta_distance) # Mean Normalization
delta_distance = delta_distance/np.linalg.norm(delta_distance) # L2-Normalization

##### Generative Capacity
total_gestures = 11
total_ids = 10
g_angle = [] # List to store Intra-Gesture Angular Capacity
g_id_angle = [] # List to store Intra-Gesture Id Angular Capacity
Capacity_Value = [] # List to store Generative Biometric Capacity of each of the gesture
d_size = 32 # Embedding Size
delta = 0 # Setting Delta(FAR Parameter) to zero

###### Iteration Loop
for gesture_val in range(total_gestures): # Iterating over the Gestures

    ###### Gesture-level
    X_store = [] # List to store all the examples of that gesture
    #idx_store = [] # List to store all the indexes of the gestures being stored
    id_store = [] # List to store the identity-labels corresponding to the gesture
    g_id_angle_store = [] # List to store Angular Spreads of the 'N' identities involved in the dataset

    ##### Gesture-Store Curation
    for g_idx, X_ges in enumerate(Test_Embeddings): # Iterating over the features

        if(y_dev[g_idx] == gesture_val): # Checking for the Gesture Labels

            X_store.append(X_ges) # Storing the Feature
            id_store.append(y_dev_id[g_idx]) # Storing ID-label of the feature

    ##### Estimation of Gesture-Level Angular Spread
    g_angle_curr,_,_ = get_cosine_bounds(np.array(X_store))
    g_angle_curr = (g_angle_curr/2)*(np.pi/180)
    g_angle.append(g_angle_curr)

    ##### Estimation of Intra-Gesture Id-Level Angular Spread
    for id_idx in range(total_ids): # Searching for Particular Identities
        X_id_store = [] # List to store Intra-Gesture features of a particular identity

        for item_idx, item in enumerate(X_store): # Iterating over the Current Gesture-Store
            
            if(id_store[item_idx] == id_idx): # Identity Match-found
                X_id_store.append(item) # Storing the Feature

        #### Estimation of Intra-Gesture Intra-Id Angular Spread
        g_id_angle_curr,_,_ = get_cosine_bounds(np.array(X_id_store))
        g_id_angle_curr = (g_id_angle_curr/2)*(np.pi/180)    
        g_id_angle_store.append(g_id_angle_curr)

    g_id_angle_curr_overall = np.average(g_id_angle_store) # Avg. Angular Spread of all the Identities within the gesture under consideration
    g_id_angle.append(g_id_angle_curr_overall) # Storing in Global List

    ##### Estimation of Gesture's Biometric Capacity
    capacity_curr = ratio_hyperspherical_caps(g_angle_curr,g_id_angle_curr_overall,1,0,d_size) # Biometric Capacity of the Current Gesture
    Capacity_Value.append(capacity_curr) # Storing Values

print(Capacity_Value)
Capacity_Value = np.array(Capacity_Value)
Capacity_Value = (Capacity_Value - np.mean(Capacity_Value))/np.std(Capacity_Value)
Capacity_Value = Capacity_Value/np.linalg.norm(Capacity_Value)

##### MasterFace
MasterFace_Capacity = Compute_MasterFace_Capacity(np.array(d_c_star),32)
print(MasterFace_Capacity)
MasterFace_Capacity = (MasterFace_Capacity - np.mean(MasterFace_Capacity))/np.std(MasterFace_Capacity)
MasterFace_Capacity = MasterFace_Capacity/np.linalg.norm(MasterFace_Capacity)

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### EER-Processing
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### EER-Processing
e = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]
e_prime = 100 - e
e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
e_prime = e_prime/np.linalg.norm(e_prime)

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### Feature-Space Scores
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### CGID and Decorr-CGID Score
C_I, C_D = CGID_Score_Calculator(Test_Embeddings,y_dev)
print('CGID Score: '+str(round(C_I,3)))              
print('CGID Score Decorrelated: '+str(round(C_D,3)))

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### Comparison Scores    
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### DGBQA-Score
###### General
beta = 0.75
alpha = 2
nu = 1
rank_dev = avg_rank_deviation(e,dgbqa_score,num_gestures)
Ar = acceptance_score(dgbqa_score,e_prime,11)
relevance = acceptance_score(dgbqa_score,e_prime,11,True)
d = pattern_match_dist(dgbqa_score,e_prime,11)
d_metric = (np.log2(2+nu*d)**(1/alpha))
O_prime = np.exp(-beta*C_D) # Orthogonal Penalty
Ar_star = Ar*d_metric
Ar_star_plusplus = Ar_star*O_prime
Ar_max = acceptance_score(dgbqa_score,e_prime,11,normalizer=True)
nAr = Ar/Ar_max
nAr_star = Ar_star/Ar_max
nAr_star_plusplus = Ar_star_plusplus/Ar_max

print('Relevance: '+str(relevance))
print('Rank Deviation: '+str(rank_dev))
print('Ar: '+str(Ar))
print('d: '+str(d))
print('d_metric: '+str(d_metric))
print('O_prime: '+str(O_prime))
print('Ar_star: '+str(Ar_star))
print('Ar_star_++: '+str(Ar_star_plusplus))
print('Ar_max: '+str(Ar_max))
print('nAr: '+str(nAr))
print('nAr_star: '+str(nAr_star))
print('nAr_star_++: '+str(nAr_star_plusplus))

###### Ranking
##### DGBQA-Score
rank_dev = avg_rank_deviation(e,dgbqa_score,num_gestures)
d = pattern_match_dist(dgbqa_score,e_prime,11)
d_metric = (np.log2(2+nu*d)**(1/alpha))
Ar_star = acceptance_score_comp(dgbqa_score,e_prime,11)*d_metric
print('======================================')
print('DGBQA-Score')
print('Rank-Deviation: '+str(rank_dev)+' Ar_star: '+str(Ar_star))

##### d_c_star
rank_dev = avg_rank_deviation(e,d_c_star,num_gestures)
d = pattern_match_dist(d_c_star,e_prime,11)
d_metric = (np.log2(2+nu*d)**(1/alpha))
Ar_star = acceptance_score_comp(d_c_star,e_prime,11)*d_metric
print('======================================')
print('d_c_star')
print('Rank-Deviation: '+str(rank_dev)+' Ar_star: '+str(Ar_star))

##### d_cs
rank_dev = avg_rank_deviation(e,d_cs,num_gestures)
d = pattern_match_dist(d_cs,e_prime,11)
d_metric = (np.log2(2+nu*d)**(1/alpha))
Ar_star = acceptance_score_comp(d_cs,e_prime,11)*d_metric
print('======================================')
print('d_cs')
print('Rank-Deviation: '+str(rank_dev)+' Ar_star: '+str(Ar_star))

##### DGBQA-Score W/o Penalty
rank_dev = avg_rank_deviation(e,dgbqa_score_wo,num_gestures)
d = pattern_match_dist(dgbqa_score_wo,e_prime,11)
d_metric = (np.log2(2+nu*d)**(1/alpha))
Ar_star = acceptance_score_comp(dgbqa_score_wo,e_prime,11)*d_metric
print('======================================')
print('DGBQA-Score W/o Penalty')
print('Rank-Deviation: '+str(rank_dev)+' Ar_star: '+str(Ar_star))

##### Delta Distance
rank_dev = avg_rank_deviation(e,delta_distance,num_gestures)
d = pattern_match_dist(delta_distance,e_prime,11)
d_metric = (np.log2(2+nu*d)**(1/alpha))
Ar_star = acceptance_score_comp(delta_distance,e_prime,11)*d_metric
print('======================================')
print('delta_distance')
print('Rank-Deviation: '+str(rank_dev)+' Ar_star: '+str(Ar_star))

##### Generative Capacity
rank_dev = avg_rank_deviation(e,Capacity_Value,num_gestures)
d = pattern_match_dist(Capacity_Value,e_prime,11)
d_metric = (np.log2(2+nu*d)**(1/alpha))
Ar_star = acceptance_score_comp(Capacity_Value,e_prime,11)*d_metric
print('======================================')
print('Generative Capacity')
print('Rank-Deviation: '+str(rank_dev)+' Ar_star: '+str(Ar_star))

##### MasterFace Capacity
rank_dev = avg_rank_deviation(e,MasterFace_Capacity,num_gestures)
d = pattern_match_dist(MasterFace_Capacity,e_prime,11)
d_metric = (np.log2(2+nu*d)**(1/alpha))
Ar_star = acceptance_score_comp(MasterFace_Capacity,e_prime,11)*d_metric
print('======================================')
print('MasterFace')
print('Rank-Deviation: '+str(rank_dev)+' Ar_star: '+str(Ar_star))
