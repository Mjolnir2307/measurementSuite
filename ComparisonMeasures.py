######## Importing libraries                                                                                                       
import numpy as np
from Levenshtein import distance

######## Comparison metrics

###### Standard metrics

##### Euclidean distances
def euclidean_distance(dgbqa_score,e_prime):

    """
    Euclidean distance between DGBQA scores and e_prime
    """
    return np.sqrt(np.sum(np.square(dgbqa_score-e_prime)))

##### Correlation
def correlation(dgbqa_score,e_prime):

    """
    Correlation between DGBQA scores and e_prime
    """

    return np.dot(dgbqa_score,e_prime)

##### DCG values
def DCG(dgbqa,e_prime):
    
    """
    Function to return DCG values
    """

    #### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx = [] # List to store arrangement orders of e_prime, with the values starting from 0, and ending at G-1
    lambda_scale = 2.0
    dcg_val = 0.0

    #### Aranging DGBQA Scores as per e_prime's order
    for idx, val in enumerate(e_prime_sort):

        for idx_search, val_search in enumerate(e_prime):

            if(val == val_search):
                arrangement_idx.append(idx_search) # Finding Index on e_prime that matches with e_prime_sort
                dgbqa_re.append(dgbqa[idx_search]) # Appending terms in dgbqa_re as per the order in e_prime_sort: The best rank is the first term
                break

    ##### DCG Estimation
    for r_e_j, dgbqa_r_e_j in enumerate(dgbqa_re): # Iterating over all the gestures in the set
        Rj = 2**(lambda_scale*dgbqa_r_e_j)
        dcg_val = dcg_val + (Rj/np.log2(2+r_e_j))

    return dcg_val

##### Kendall's tau distance
def kendall_tau(dgbqa,e_prime,G_total):

    """
    Function to return Kendall's Tau distance values
    """

    #### String joining
    def join_str(ip,G_total):
        op = ''
        for g_idx in range(G_total):
            op = op + str(ip[g_idx])
        return op

    #### Defining Essentials
    e_prime_sort = -(np.sort(-e_prime)) # Sorted e_prime
    dgbqa_sort = -(np.sort(-dgbqa)) # Sorted DGBQA-Score Values
    dgbqa_re = [] # DGBQA-Scores Ordered as per the e_prime_sort  
    arrangement_idx_e = [] # List to store arrangement orders of e_prime_sort
    arrangement_idx_d = [] # List to store arrangement orders of dgbqa_sort

    #### Arrangement index: e_prime
    for idx, val in enumerate(e_prime_sort):
        for idx_search, val_search in enumerate(e_prime):
            if(val == val_search):
                arrangement_idx_e.append(idx_search)
                dgbqa_re.append(dgbqa[idx_search])
                break
    op_e = join_str(arrangement_idx_e,G_total)

    #### Arrangement index: DGBQA
    for idx, val in enumerate(dgbqa_re): 
        for idx_search, val_search in enumerate(dgbqa_sort):
            if(val == val_search):
                arrangement_idx_d.append(idx_search)
                break
    op_d = join_str(arrangement_idx_d,G_total)
    
    return distance(op_d,op_e)
