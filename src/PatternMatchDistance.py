######## Importing libraries
import os                                                                                                         
import gc
import math
import numpy as np

####### Pattern Match Distance
def pattern_match_dist(dgbqa,e_prime,G):

    """
    Function to compute Pattern-Match Distance

    INPUTS:-
    1) dgbqa: Array of unranked but normalized DGBQA score values
    2) e_prime: Array of unranked but normalized (1 - EER) values
    3) G: Total number of gestures considered for analysis

    OUTPUTS:-
    1) pm_dist: Pattern-Match Distance
    """
    ##### Defining Essentials
    e_prime_sort = -np.sort(-np.array(e_prime)) # Sorting e_prime
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1
    pm_dist = 0 # Initializing pm_dist as 0
    dgbqa_re_f = [] # Low-to-High Ranked List of DGBQA-Score sorted as per e_prime  
    e_prime_sort_f = [] # Low-to-High Ranked List of e_prime values

    ##### e_prime in reverse order
    for idx in range(len(e_prime_sort)):
        e_prime_sort_f.append(e_prime_sort[len(e_prime_sort)-idx-1])

    e_prime_sort_f = np.array(e_prime_sort_f) # Array Formation

    ##### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break

    dgbqa_re_b = np.array(dgbqa_re) # Array Formation
    
    ##### Aranging DGBQA Scores as per e_prime's order in Low to High
    for idx in range(len(dgbqa_re)):
        dgbqa_re_f.append(dgbqa_re[len(dgbqa_re)-idx-1])

    dgbqa_re_f = np.array(dgbqa_re_f) # Array Formation 

    ##### Pattern-Matching Distance Estimation

    #### Slope Computation
    tan_theta_f = dgbqa_re_f[1:] - dgbqa_re_f[:-1] # Forward Movement Slopes
    tan_theta_b = tan_theta_f # Backward Movement Slope is Similar as the Forward Movement Slope, just estimation method differs

    #### Forward Value Computation
    ### Estimation
    e_bar_f = e_prime_sort_f[:-1] + tan_theta_f # e2_bar_f to eG_bar_f
    eG_bar_f = e_bar_f[-1] # Extracting Last Value
    e_bar_f = e_bar_f[:-1] # Storing Middle 'G-2' Values

    ### Error Computation
    error_f = np.sum(np.abs(e_bar_f - e_prime_sort_f[1:-1])) + 2*(np.abs(eG_bar_f - e_prime_sort_f[-1]))

    #### Backward Value Computation
    ### Estimation
    e_bar_b = e_prime_sort_f[1:] - tan_theta_f # e1_bar to e(G-1)_bar_f
    e1_bar_b = e_bar_b[0] # Extracting First Value
    e_bar_b = e_bar_b[1:] # Storing Middle 'G-2' Values

    ### Error Compuation
    error_b = np.sum(np.abs(e_bar_b - e_prime_sort_f[1:-1])) + 2*(np.abs(e1_bar_b - e_prime_sort_f[0])) 
    
    #### Pattern-Matching Distance
    pm_dist = 0.5*(error_f + error_b)

    return pm_dist