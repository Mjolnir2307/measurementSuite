####### Importing Libraries
import os
import argparse
import numpy as np
import tensorflow as tf
from src.DGBQA_Score import gbqa_delta_dist_compute
from src.ICGDScore import CGID_Score_Calculator
from src.RankDeviation import avg_rank_deviation
from src.AcceptanceScore import acceptance_score
from src.PatternMatchDistance import pattern_match_dist

####### Model selection
def get_val(embedding,
            y_dev,
            y_dev_id,
            eer_values,
            G_total,
            I_total,
            measure_req,
            mode):
    
    """
    Function to seek a particular measure
    """

    ##### DGBQA score computation
    dgbqa_score = [] # DGBQA Score
    for g_id in range(G_total):
        dgbqa_score_curr, _, _, _ = gbqa_delta_dist_compute(embedding,g_id,I_total,y_dev,y_dev_id)
        dgbqa_score.append(dgbqa_score_curr)

    dgbqa_score = np.array(dgbqa_score) # Array Formation
    dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
    dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

    ##### Ground truth equal error rates
    e_prime = 100 - np.array(eer_values)
    e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
    e_prime = e_prime/np.linalg.norm(e_prime)

    ##### Metric computation
    if(mode == 'single'): # Only one metric is required:
        if(measure_req == 'r'): # Rank deviation
            return avg_rank_deviation(np.array(eer_values),
                                    dgbqa_score,
                                    G_total)
        
        if(measure_req == 'R'): # Relevance
            return acceptance_score(dgbqa_score,
                                    e_prime,
                                    G_total,
                                    False,
                                    True)
        
        if(measure_req == 'psi'): # Pattern-match distance
            return pattern_match_dist(dgbqa_score,
                                    e_prime,
                                    G_total)
        
        if(measure_req == 'Cd'): # ICGD score
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            return C_D
        
        if(measure_req == 'Ar'): # Acceptance score
            return acceptance_score(dgbqa_score,
                                    e_prime,
                                    G_total,
                                    False,
                                    False)
        
        if(measure_req == 'ArCd'): # Ar*C_D
            beta = 0.75
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            ArCd = acceptance_score(dgbqa_score,e_prime,G_total,False,False)* np.exp(-beta*C_D)
            return ArCd
        
        if(measure_req == 'Ar_psi'): # Ar*psi
            nu = 1
            alpha = 2
            d = pattern_match_dist(dgbqa_score,e_prime,G_total)
            return acceptance_score(dgbqa_score,e_prime,G_total,False,False)*(np.log2(2+nu*d)**(-1/alpha))
        
        if(measure_req == 'Cd_psi'): # C_D*psi
            nu = 1
            alpha = 2
            beta = 0.75
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            d = pattern_match_dist(dgbqa_score,e_prime,G_total)
            return (np.log2(2+nu*d)**(-1/alpha))*np.exp(-beta*C_D)
        
        if(measure_req == 'Ar*'): # Full: Ar* x psi x C_D
            nu = 1
            alpha = 2
            beta = 0.75
            C_I, C_D = CGID_Score_Calculator(embedding,y_dev)
            d = pattern_match_dist(dgbqa_score,e_prime,G_total)
            return acceptance_score(dgbqa_score,e_prime,G_total,False,False)*(np.log2(2+nu*d)**(-1/alpha))*np.exp(-beta*C_D)
        
    if(mode == 'full'): # returns all nine metrics
        nu = 1
        alpha = 2
        beta = 0.75

        r = avg_rank_deviation(np.array(eer_values),
                                    dgbqa_score,
                                    G_total) # rank deviation
        R = acceptance_score(dgbqa_score,
                                    e_prime,
                                    G_total,
                                    False,
                                    True) # Relevance
        d = pattern_match_dist(dgbqa_score,
                                    e_prime,
                                    G_total) # Pattern match distance
        C_I, C_D = CGID_Score_Calculator(embedding,y_dev) # ICGD score
        Ar = acceptance_score(dgbqa_score,
                                    e_prime,
                                    G_total,
                                    False,
                                    False) # Ar
        ArCd = Ar* np.exp(-beta*C_D) # Ar*C_D
        Ar_psi = Ar*(np.log2(2+nu*d)**(-1/alpha)) # Ar*psi
        Cd_psi = (np.log2(2+nu*d)**(-1/alpha))*np.exp(-beta*C_D) # Cd*psi
        Ar_star = Ar*(np.log2(2+nu*d)**(-1/alpha))* np.exp(-beta*C_D) # Ar*
        return [r,
                R,
                d,
                C_D,
                Ar,
                ArCd,
                Ar_psi,
                Cd_psi,
                Ar_star] # List of measures
    
def select_model(embedding_list,
                 dataset_list,
                 var,
                 ):
    
    """
    Function to get optimal model as per metrics
    
    INPUTS:-
    1) embedding_list: The list of embeddings from which the optimal is to be derived
    2) dataset_list: Corresponding list of the dataset
    3) var: The measure upon which optimal is to be derived

    OUPUTS:-
    1) opt_model: The optimal model/models
    """

    measure_val = [] # Value store

    ##### Iteration over embeddings
    for idx_curr, embedding in enumerate(embedding_list):

        embedding_curr = np.load(embedding,allow_pickle=True)['arr_0']
        dataset_curr = dataset_list[idx_curr] # Current dataset

        if(dataset_curr == 'Soli'):
            y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']
            G_total = 11
            I_total = 10
            eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]

        if(dataset_curr == 'HandLogin'):
            y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_HandLogin.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz')['arr_0']
            G_total = 4
            I_total = 16
            eer_values = [0.44,1.29,4.89,1.05]

        if(dataset_curr == 'Tiny'):
            y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_Tiny.npz')['arr_0']
            y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz')['arr_0']
            G_total = 11
            I_total = 26

            e1_val = 100 - 16.45
            e2_val = 100 - 23.36 
            e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
            e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])
            eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)
            eer_values = list(eer_values)

        ##### Measure computation
        val_curr = get_val(embedding_curr,
                           y_dev,
                           y_dev_id,
                           eer_values,
                           G_total,
                           I_total,
                           var,
                           'single') # Current value
        measure_val.append(val_curr)

    ##### Optimal selection
    print(measure_val)
    if(var in ['R','Ar','ArCd','Ar_psi','Cd_psi','Ar*']):
        opt_model = embedding_list[int(np.argmax(measure_val))]
    else:
        opt_model = embedding_list[int(np.argmin(measure_val))]

    return opt_model

###### Testing
opt_model = select_model(embedding_list=['./Embeddings/MS_MViT_pt5-pt5_SOLI.npz',
                                         './Embeddings/MS_MViT_pt5-1_SOLI.npz',
                                         './Embeddings/MS_MViT_pt5-1pt5_SOLI.npz',
                                         './Embeddings/MS_MViT_1-pt5_SOLI.npz',
                                         './Embeddings/MS_MViT_1-1_SOLI.npz',
                                         './Embeddings/MS_MViT_1-1pt5_SOLI.npz',
                                         './Embeddings/MS_MViT_1pt5-pt5_SOLI.npz',
                                         './Embeddings/MS_MViT_1pt5-1_SOLI.npz',
                                         './Embeddings/MS_MViT_1pt5-1pt5_SOLI.npz',],
                        dataset_list=['Soli',
                                      'Soli',
                                      'Soli',
                                      'Soli',
                                      'Soli',
                                      'Soli',
                                      'Soli',
                                      'Soli',
                                      'Soli'],
                        var='R')
print(opt_model)
