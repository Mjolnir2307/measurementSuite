######## Importing libraries
import os                                                                                                         
import gc
import math
import numpy as np
import tensorflow as tf

###### CGID and Decorrelated-CGID Score

##### Mask Generation
#### Positive Mask
def get_positive_mask(labels):
        """
        Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = tf.cast(tf.eye(labels.shape[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        # Combine the two masks``
        mask = tf.logical_and(indices_not_equal, labels_equal)

        return mask
    
###### Negative Mask - Different Mask
def get_negative_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

####### CGID Score
def CGID_Score_Calculator(f_theta,y_hgr):
      
    """
    Function to Compute CGID Score and Plot correponding Gramian Matrix Heatmap
          
    INPUTS:-
    1) f_theta: Embeddings of the shape (N,d); N-Total Examples, d: Embedding Dimensions
    2) y_hgr: HGR labels of the shape (N,)
    3) filepath: Path to save the plotted 
    
    OUTPUTS:-
    1) cgid_score_decorr: Average of gram matrix computed over the all the examples
    2) cgid_score: Average of masked gram matrix
    """

    ##### L2-Normalization
    f_theta = tf.math.l2_normalize(f_theta,axis=1)

    ##### Different Gesture Mask Computation
    delta_bar = get_negative_mask(y_hgr)
    delta_bar = np.reshape(delta_bar.numpy(),(delta_bar.shape[0],delta_bar.shape[1]))

    ##### Gramian Matrix Formation
    G_bar = tf.cast(tf.linalg.matmul(f_theta,f_theta,transpose_b=True),dtype=tf.float32)
    cgid_score_matrix = np.multiply(delta_bar,G_bar.numpy())
    cgid_score_matrix_mask = (cgid_score_matrix >= 0)
    #plot_heatmap(cgid_score_matrix*cgid_score_matrix_mask,filepath)

    ##### CGID-Score
    cgid_score = (np.sum(cgid_score_matrix*cgid_score_matrix_mask))/(np.sum(cgid_score_matrix_mask))
    cgid_score_decorr = np.mean(cgid_score_matrix)
    
    return cgid_score_decorr, cgid_score