####### Importing libraries
import numpy as np

####### MasterFace Capacity
def Compute_MasterFace_Capacity(d_c_star,d_size): 

    """
    Function to Compute MasterFace based Gesture Capacity

    INPUTS:-
    1) d_c_star: Avg. Distance between different Identity Centroids within a Gesture
    2) d_size: Size of the Embeddings

    OUTPUTS:-
    1) MasterFace_Capacity: Estimated Biometric Capacity within Hand-Gestures  
    """

    MasterFace_Capacity = np.exp(d_size*(0.993 - 0.436*(1/d_c_star))+3.701-3.706*(1/d_c_star))
    return MasterFace_Capacity