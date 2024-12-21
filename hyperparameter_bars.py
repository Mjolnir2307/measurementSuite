####### Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from hyperparameter import get_nar

####### Testing
embedding_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz',
                  './Embeddings/DGBQA_CGID_MF_1pt5-pt5_SOLI.npz'
                  './Embeddings/MS_TPN_pt5-pt5_SOLI.npz',
                  './Embeddings/MS_TAM_1-pt5_SOLI.npz',
                  './Embeddings/MS_MViT_pt5-1_SOLI.npz']
y_dev_path = './Embeddings/y_dev_DeltaDistance_SOLI.npz'
y_dev_id_path = './Embeddings/y_dev_id_DeltaDistance_SOLI.npz'
hyp_val = [0.25,0.50,0.75,1.00,2.00,4.00]
eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]
nar_values = []

for hyp_val_curr in hyp_val:
    nar_values.append(get_nar(embedding_list,
                  y_dev_path,
                  y_dev_id_path,
                  eer_values,
                  11,
                  10,
                  alpha=2,
                  beta=0.75,
                  lambda_scale=2,
                  kappa=hyp_val_curr,
                  nu=1))
    
nar_values = np.array(nar_values) # shape -> (num_hyp,num_models)

plt.plot(hyp_val,nar_values[:,0],label='Res3D-ViViT',linewidth=4,linestyle='dashdot')
plt.plot(hyp_val,nar_values[:,1],label='Res3D-MF',linewidth=4,linestyle='dashdot')
plt.plot(hyp_val,nar_values[:,2],label='Res3D-TPN',linewidth=4,linestyle='dashdot')
plt.plot(hyp_val,nar_values[:,3],label='Res3D-TAM',linewidth=4,linestyle='dashdot')
plt.plot(hyp_val,nar_values[:,4],label='Res3D-MF',linewidth=4,linestyle='dashdot')
plt.show()

####### Curve Plotting
#fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(nrows=1,
#                                      ncols=4,
#                                      figsize=(12,8))


