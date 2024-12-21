####### Importing Libraries
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.ticker import LinearLocator
from hyperparameter import compute_acceptance

####### Surface Plot
def get_nar(embedding_list,
                 y_dev_path, 
                 y_dev_id_path, 
                 eer_values, 
                 G_total,
                 I_total,
                 alpha,
                 beta,
                 lambda_scale,
                 kappa,
                 nu):
    
    """
    Function to plot surface plot wrt lambda values 
    """

    y_dev = np.load(y_dev_path)['arr_0']
    y_dev_id = np.load(y_dev_id_path)['arr_0']
    nar_value = []

    for embedding_path in embedding_list:
        embedding = np.load(embedding_path)['arr_0']
        nar_value.append(compute_acceptance(embedding,
                                            y_dev,
                                            y_dev_id,
                                            eer_values,
                                            G_total,
                                            I_total,
                                            alpha,
                                            beta,
                                            lambda_scale,
                                            kappa,
                                            nu))
    return np.array(nar_value)
        
###### Surface Plot
def plot_surface(nar_val):

    #print(nar_val)
    ax = plt.figure().add_subplot(projection='3d')
    
    X = Y = np.arange(0.5,2.0,0.5)
    X, Y = np.meshgrid(X,Y)
    Z = np.array(nar_val).reshape(X.shape)
    #print(X,Y,Z)
    ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
    ax.plot_surface(Y,X,Z,alpha=0.3)
    ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
    
    X = np.array([0.5,0.5,0.5,1.0,1.0,1.0,1.5,1.5,1.5])
    Y = np.array([0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5])
    Z = np.array(nar_val)

    ax.scatter(X,Y,Z,
               s=200,
               color='black',
               edgecolors='k',
               marker='h')
    #ax.stem(X,Y,Z)

    ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
    ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
    ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)

    ax.view_init(elev=17., azim=30) 
    ax.set_zlim(0.05,0.55)
    plt.show()

###### Defining essentials
embedding_list = ['./Embeddings/MS_MViT_pt5-pt5_SOLI.npz',
                  './Embeddings/MS_MViT_pt5-1_SOLI.npz',
                  './Embeddings/MS_MViT_pt5-1pt5_SOLI.npz',
                  './Embeddings/MS_MViT_1-pt5_SOLI.npz',
                  './Embeddings/MS_MViT_1-1_SOLI.npz',
                  './Embeddings/MS_MViT_1-1pt5_SOLI.npz',
                  './Embeddings/MS_MViT_1pt5-pt5_SOLI.npz',
                  './Embeddings/MS_MViT_1pt5-1_SOLI.npz',
                  './Embeddings/MS_MViT_1pt5-1pt5_SOLI.npz']
y_dev_path = './Embeddings/y_dev_DeltaDistance_SOLI.npz'
y_dev_id_path = './Embeddings/y_dev_id_DeltaDistance_SOLI.npz'
eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]
G_total=11
I_total=10
X = Y = np.arange(0.5,2.0,0.5)
X, Y = np.meshgrid(X,Y)
X_scatter = np.array([0.5,0.5,0.5,1.0,1.0,1.0,1.5,1.5,1.5])
Y_scatter = np.array([0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5])
#Z = np.array(nar_val).reshape(X.shape)
#plot_surface(get_nar(embedding_list,
#                     y_dev_path,
#                     y_dev_id_path,
#                     eer_values,
#                     11,
#                     10,
#                     alpha=2,
#                     beta=0.75,
#                     lambda_scale=2,
#                     kappa=1,
#                    nu=1))

###### Plotting
fig, ((ax11, ax12, ax13, ax14, ax15, ax16), 
      (ax21, ax22, ax23, ax24, ax25, ax26),
      (ax31, ax32, ax33, ax34, ax35, ax36),
      (ax41, ax42, ax43, ax44, ax45, ax46)) = plt.subplots(nrows=4, ncols=6, figsize=(18,16))

for ax in [ax11, ax12, ax13, ax14, ax15, ax16,
           ax21, ax22, ax23, ax24, ax25, ax26,
           ax31, ax32, ax33, ax34, ax35, ax36,
           ax41, ax42, ax43, ax44, ax45, ax46]:
    
    ##### Kappa
    
    #### K=1/4
    if(ax==ax11):
        nar_val = get_nar(embedding_list,y_dev_path,y_dev_id_path,
                     eer_values,
                     11,10,
                     alpha=2,
                     beta=0.75,
                     lambda_scale=2,
                     kappa=1/4,
                    nu=1)
        Z = np.array(nar_val).reshape(X.shape)
        ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
        ax.plot_surface(Y,X,Z,alpha=0.3)
        ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
        Z = np.array(nar_val)
        ax.scatter(X,Y,Z,
                s=200,
                color='black',
                edgecolors='k',
                marker='h')
        ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
        ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
        ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)

        ax.view_init(elev=17., azim=30) 
        ax.set_zlim(0.05,0.55)

    #### K=1/2
    if(ax==ax11):
        nar_val = get_nar(embedding_list,y_dev_path,y_dev_id_path,
                     eer_values,
                     11,10,
                     alpha=2,
                     beta=0.75,
                     lambda_scale=2,
                     kappa=1/2,
                    nu=1)
        Z = np.array(nar_val).reshape(X.shape)
        ax.plot_wireframe(Y,X,Z,
                    lw=2.0)
        ax.plot_surface(Y,X,Z,alpha=0.3)
        ax.contourf(Y, X, Z, zdir='z', offset=0.05, cmap='Blues')
        Z = np.array(nar_val)
        ax.scatter(X,Y,Z,
                s=200,
                color='black',
                edgecolors='k',
                marker='h')
        ax.set_zlabel('$nAr^{*}(\Delta)$',fontsize=12)
        ax.set_xlabel('$\lambda_{ID}$',fontsize=12) 
        ax.set_ylabel('$\lambda_{ICGD}$',fontsize=12)

        ax.view_init(elev=17., azim=30) 
        ax.set_zlim(0.05,0.55)



