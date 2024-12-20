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

####### Generative Capacity
###### Cosine Bounds Function
def get_cosine_bounds(X, quantile=0.05):
    cosine_dist = np.dot(X, X.transpose())
    min_val = np.min(cosine_dist, axis=1)

    mask = np.ones(cosine_dist.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    cosine_dist = cosine_dist * mask

    max_val = np.max(cosine_dist, axis=1)
    range_val = max_val - min_val

    value = np.quantile(min_val, quantile)
    total_angle = np.arccos(value) * 180 / np.pi

    return total_angle, min_val, max_val

###### Capacity Estimation Function
def ratio_hyperspherical_caps(inter_class_angle, intra_class_angle, cos_delta, sin_delta, d):
    
    # compute cos(\theta) where \theta is the solid
    # angle corresponding to the inter-class hyper-spherical cap
    cos_theta = np.cos(inter_class_angle)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    cos_omega = cos_theta * cos_delta - sin_theta * sin_delta
     
    x = 1 - cos_omega * cos_omega
    a = (d - 1)/2
    b = 0.5 

    numerator = sp.betainc(a, b, x)
    index = np.where(cos_omega < 0)
    #numerator[index] = 0.5 + numerator[index]
    
    # compute cos(\phi) where \phi is the solid
    # angle corresponding to the intra-class hyper-spherical cap
    cos_phi = np.cos(intra_class_angle)
    sin_phi = np.sqrt(1 - cos_phi**2)
    cos_omega = cos_phi * cos_delta - sin_phi * sin_delta

    x = 1 - cos_omega * cos_omega
    a = (d - 1)/2
    b = 0.5

    denominator = sp.betainc(a, b, x)
    index = np.where(cos_omega < 0)
    #denominator[index] = 0.5 + denominator[index]

    capacity = numerator / denominator
    #capacity[capacity < 1] = 1
    
    return capacity