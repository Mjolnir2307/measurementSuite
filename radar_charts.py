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

e1_val = 100 - 16.45
e2_val = 100 - 23.36 
e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])
eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)

embedding_list_full = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1pt5_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-2pt5_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1pt5_Tiny.npz',
                       './Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz',
                       './Embeddings/MS_MF_1-1_Tiny.npz',
                       './Embeddings/MS_MF_1-1pt5_Tiny.npz',
                       './Embeddings/MS_MF_1-2pt5_Tiny.npz',
                       './Embeddings/MS_TAM_1-1_Tiny.npz',
                       './Embeddings/MS_TAM_1-1pt5_Tiny.npz',
                       './Embeddings/MS_TAM_1-2pt5_Tiny.npz']

dataset_list = ['Tiny']*12
y_dev_path = './Embeddings/y_dev_DGBQA_Seen_Tiny.npz'
y_dev_id_path = './Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz'

model_list = []

#for measure in ['euclid','corr','DCG','Kendall','ERR','U','GRE','infAp','NegRel','RPP','Ar*']:
#    model_list.append(select_model(embedding_list_full,
#                                   dataset_list,
#                                   measure
#                                   ))
#print(model_list)

#model_list = ['./Embeddings/MS_TPN_1pt5-1_SOLI.npz', './Embeddings/MS_TPN_1pt5-1_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-1_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-MF_1-1_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-MF_1-1pt5_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1pt5_SOLI.npz', './Embeddings/MS_MViT_pt5-1_SOLI.npz', './Embeddings/MS_MViT_pt5-1_SOLI.npz']
model_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz', './Embeddings/MS_MF_1-1_Tiny.npz', './Embeddings/MS_MF_1-1pt5_Tiny.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz', './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-2pt5_Tiny.npz', './Embeddings/MS_MF_1-1pt5_Tiny.npz']
data = get_data(model_list,
                y_dev_path,
                y_dev_id_path,
                eer_values,
                11,
                26,
                alpha=2,
                beta=0.75,
                lambda_scale=2,
                kappa=1,
                nu=1,
                num_entries=11)
N = 4
theta = radar_factory(N, frame='polygon')
colors = ['tab:blue',
          'tab:orange',
          'tab:green',
          'gold',
          'tab:purple',
          'tab:brown',
          'tab:pink',
          'tab:gray',
          'tab:olive',
          'tab:cyan',
          'tab:red']
spoke_labels = data.pop(0)
#print(data)

fig, ax = plt.subplots(figsize=(9,8), nrows=1, ncols=1,
                        subplot_kw=dict(projection='radar'))

for d, color in zip(data, colors):
    ax.plot(theta,d,color=color,marker='o',markersize=10)
    ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
ax.set_varlabels(spoke_labels)
ax.set_xlabel('(c) TinyRadar',fontsize=14)

labels = ('RMSE:'+' Res3D-ViViT(1.0,2.5)',
          'Cosine similarity:'+' Res3D-ViViT(1.0,2.5)',
          'DCG:'+' Res3D-ViViT(0.5,1.0)',
          'Kendall:'+' Res3D-ViViT(0.5,1.0)',
          'ERR:'+' Res3D-ViViT(0.5,1.0)',
          'U-measure:'+' Res3D-ViViT(1.0,2.5)',
          'GRE:'+' Res3D-MF(1.0,1.0)',
          'InfAp:'+' Res3D-MF(1.0,1.5)',
          'Negative relevance:'+' Res3D-ViViT(0.5,1.0)',
          'RPP:'+' Res3D-ViViT(0.5,2.5)',
          '$nAr^{*}(\Delta):$'+' Res3D-MF(1.0,1.5)'
          )

legend = ax.legend(labels,
                   loc=(0.75,0.75),
                   labelspacing=0.1, 
                   fontsize='10')

plt.show()

