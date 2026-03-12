# biomQuant

A package consisting of evaluation measures for gesture biometric quantification.

This package provide four measures:

1. Rank deviation ($\hat{r}$)
2. Relevance ($\mathcal{R}$)
3. Trend match distance ($\Psi$)
4. ICGD Score ($C_d$)

We combine these in advanced acceptance score

${A_r}^*~=~\frac{\sum_{j=1}^{G}\Bigl( \frac{2^{\lambda\mathcal{R}_j}}{\exp(\kappa*(r_{j}^{\Delta} - r_{j}^{\hat{e}} ))}\Bigr)}{\sqrt{\log_2(2+\nu\Psi)}}*{\exp(-\beta C_d)}$

Here, $G$ is the number of gestures. While $r_{j}^{\Delta}$ and $r_{j}^{\hat{e}}$ denote ranks of the $j^{th}$ gesture wrt the estimated biometric estimates and the ground truth respectively. $\lambda,~\kappa,~\nu,~\text{and}~\beta$ are the scaling factors.

We further normalize this into $nA_r^*(\Delta)$. Mathematically, 

$nA_r^*(\Delta)=\frac{A_r^{*}(\Delta)}{A_r^{*}(\hat{e})}$

Where, $A_r^*(\Delta)$ and $A_r^*(\hat{e})$ represents $A_r^*$ values for the output DGBQA scores and ground truth, i.e.,

$A_r^*(\hat{e})=\sum_{j=1}^{G}2^{\lambda \Bigl[ \gamma\Bigl({\frac{G-{r_{j}^{\hat{e}}}+1}{G}\Bigr){\hat{e}\left[r_{j}^{\hat{e}}\right]}}+\Bigl(\frac{r_{j}^{\hat{e}}}{G}\Bigr)\Bigl(1-\hat{e}\left[r_{j}^{\hat{e}}\right]\Bigr) \Bigr]}$

## Requirements

1. numpy
2. sckit-learn
3. scipy
4. tensorflow $\geq$ 2.8.0

## How to use

1. Advanced Acceptance Score

```python
from biomQaunt.advancedAcceptance import comp_advancedAcceptance
nAr_star = comp_advancedAcceptance(biometricParams,
                                   groundTruth,
                                   embeddings,
                                   labels,
                                   G=numGestures)
```

2. Rank deviation

```python
from biomQaunt.rankDev import rankDev
r_prime = rankDev(1-groundTruth,
                  biometricParams,
                  G=numGestures)
```

3. Relevance

```python
import numpy as np
from biomQaunt.acceptanceScore import compAr

def preProcess(inputVec):
    inputVec = (inputVec - np.mean(inputVec))/np.std(inputVec)
    return inputVec/np.linalg.norm(inputVec)

relevance = compAr(preProcess(biometricParams),
                   preProcess(groundTruth),
                   normalizer=False,
                   relevance=True)
```

4. ICGD score

```python
import tensorflow as tf
from biomQaunt.icgd import compICGD

def normalisation_layer(x):   
    return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))

embeddings = tf.keras.layers.Lambda(normalisation_layer)(embeddings)

icgdScore = icgdScore(embeddings.numpy(),
                     labels)
```

5. Trend match distance

```python
from biomQaunt.trendMatch import compTrendMatchDist
psi = rankDev(biometricParams,
              groundTruth,
              G=numGestures)
```
