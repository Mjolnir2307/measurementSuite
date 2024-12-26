####### Importing Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from RadarPlot import get_data, radar_factory
from ModelSelector import select_model

####### Plotting
alpha = 2
beta = 0.75
kappa = 1
lambda_scale = 2
nu = 1

eer_values = np.array([15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63])
embedding_list_full = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-1pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_pt5-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_pt5-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_pt5-1pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_1-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_1-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_1-1pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-1_SOLI.npz',
                       './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-1pt5_SOLI.npz',
                       './Embeddings/MS_TPN_pt5-pt5_SOLI.npz',
                       './Embeddings/MS_TPN_pt5-1_SOLI.npz',
                       './Embeddings/MS_TPN_pt5-1pt5_SOLI.npz',
                       './Embeddings/MS_TPN_1-pt5_SOLI.npz',
                       './Embeddings/MS_TPN_1-1_SOLI.npz',
                       './Embeddings/MS_TPN_1-1pt5_SOLI.npz',
                       './Embeddings/MS_TPN_1pt5-pt5_SOLI.npz',
                       './Embeddings/MS_TPN_1pt5-1_SOLI.npz',
                       './Embeddings/MS_TPN_1pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_TAM_pt5-pt5_SOLI.npz',
                       './Embeddings/MS_TAM_1-pt5_SOLI.npz',
                       './Embeddings/MS_TAM_1-1_SOLI.npz',
                       './Embeddings/MS_MViT_pt5-pt5_SOLI.npz',
                       './Embeddings/MS_MViT_pt5-1_SOLI.npz',
                       './Embeddings/MS_MViT_pt5-1pt5_SOLI.npz',
                       './Embeddings/MS_MViT_1-pt5_SOLI.npz',
                       './Embeddings/MS_MViT_1-1_SOLI.npz',
                       './Embeddings/MS_MViT_1-1pt5_SOLI.npz',
                       './Embeddings/MS_MViT_1pt5-pt5_SOLI.npz',
                       './Embeddings/MS_MViT_1pt5-1_SOLI.npz',
                       './Embeddings/MS_MViT_1pt5-1pt5_SOLI.npz']

dataset_list = ['Soli']*39
y_dev_path = './Embeddings/y_dev_DeltaDistance_SOLI.npz'
y_dev_id_path = './Embeddings/y_dev_id_DeltaDistance_SOLI.npz'

model_list = []

#for measure in ['r','R','psi','Cd','Ar','ArCd','Ar_psi','Cd_psi','Ar*']:
#    model_list.append(select_model(embedding_list_full,
#                                   dataset_list,
#                                   measure
#                                   ))
#print(model_list)

model_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1-pt5_SOLI.npz', './Embeddings/MS_TPN_1-1pt5_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz', './Embeddings/MS_MViT_pt5-1_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz', './Embeddings/MS_MViT_pt5-1_SOLI.npz', './Embeddings/MS_MViT_pt5-1_SOLI.npz']

data = get_data(model_list,
                y_dev_path,
                y_dev_id_path,
                eer_values,
                11,
                10,
                alpha=2,
                beta=0.75,
                lambda_scale=2,
                kappa=1,
                nu=1,
                num_entries=9)
N = 4
theta = radar_factory(N, frame='polygon')
colors = ['b', 'r', 'g', 'm', 'y', '#9CDADB', '#FF00DE', '#FF9900', 'cyan']
spoke_labels = data.pop(0)
#print(data)

fig, ax = plt.subplots(figsize=(9,8), nrows=1, ncols=1,
                        subplot_kw=dict(projection='radar'))

for d, color in zip(data, colors):
    ax.plot(theta,d,color=color,marker='o',markersize=10)
    ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
ax.set_varlabels(spoke_labels)
ax.set_xlabel('(a) Soli',fontsize=14)

labels = ('$\\mathcal{r}$:'+' Res3D-ViViT(1.5,0.5)',
          'R:'+' Res3D-MF(1.5,0.5)',
          '$\\psi$:'+' Res3D-ViViT(1.0,0.5)',
          '$C_{d}$:'+' Res3D-TPN(1.0,1.5)',
          '$Ar(\Delta)$:'+' Res3D-ViViT(1.5,0.5)',
          '$Ar(\Delta)*\bar{O}$:'+' Res3D-MViT(0.5,1.0)',
          '$Ar(\Delta)*\\bar{\psi}:$'+' Res3D-ViViT(1.5,0.5)',
          '$\\bar{\psi}*\\bar{O}:$'+' Res3D-MViT(0.5,1.0)',
          '$nAr^{*}(\Delta):$'+' Res3D-MViT(0.5,1.0)'
          )

legend = ax.legend(labels,
                   loc=(0.75,0.75),
                   labelspacing=0.1, 
                   fontsize='10')

plt.show()

