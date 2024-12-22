######## Importing libraries
import os                                                                                                         
import gc
import math
import numpy as np
from src.AcceptanceScore import rank_compute_acc

def acceptance_score_comp(dgbqa,e_prime,G):
 
    """
    Function to compute Acceptance Score: Sum over all Gestures(Relevance/Rank Deviation)

    INPUTS:-
    1) dgbqa: Array of unranked but normalized DGBQA score values
    2) e_prime: Array of unranked but normalized (1 - EER) values
    3) G: Total number of gestures considered for analysis

    OUTPUTS:-
    1) Ar: Acceptance Score: Sum over all Gestures(Relevance/Rank Deviation)
    """

    ##### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1
    Ar = 0 # Initializing Acceptance Value as zero
    lambda_scale = 2 # Scaling factor for relevance
    gamma = 2 # Scaling Factor for the first term in relevance formulation
    kappa = 1  # Scaling Factor for the Rank-Deviation Penalty

    ##### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break

        #print(arrangement_idx)

    ##### Ar Estimation
    for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set

        #### Relevance Gain
        #R_j = gamma*((G-(r_e_j+1)+1)/G)*dgbqa_r_e_j + ((r_e_j+1)/G)*(1 - dgbqa_r_e_j)
        #R_j = 2**(lambda_scale*R_j)

        #### Rank Deviation Penalty
        ### Rank Computation
        rank_dgbqa = rank_compute_acc(np.array(dgbqa_sort),dgbqa_r_e_j) # Rank Derived as per DGBQA-Score Estimates
        rank_e_prime = rank_compute_acc(np.array(dgbqa_re),dgbqa_r_e_j) # Rank Derived as per e-prime-sort based sorting of DGBQA-Scores

        ### Rank Deviation
        rank_dev_j = np.exp(kappa*np.abs(rank_dgbqa - rank_e_prime)) 

        #### Ar Estimate for the Current Gesture
        Ar_j = 1/rank_dev_j
        Ar = Ar + Ar_j # Adding this to the Metric
        #print(r_e_j,R_j,np.abs(rank_dgbqa - rank_e_prime),rank_dev_j,Ar_j)1

    return Ar