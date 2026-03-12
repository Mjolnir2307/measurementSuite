# Advanced Acceptane Score: A Holistic Measure for Gesture Biometric Quantification

![alt text](https://github.com/Mjolnir2307/measurementSuite/blob/main/MS_Comp_Radar.jpg "Comparison of the proposed $A_{r}^{*}$ with existing measures. The figure highlights that our proposed measures fulfills all the design criteria optimally.")

Abstract: Quantifying biometric characteristics of hand gestures is essential for discovering novel biometric traits. It involves arriving at fitness scores from a gesture- and identity-aware feature space. However, evaluating the quality of these scores remains an open problem. Existing biometric literature relies on assessing per-sample error rates which requires gestures to be categorized with respect to biometric characteristics and thus are not compatible with the estimated biometric characteristics. In this work, we present an exhaustive set of task-specific evaluation measures. First, we identify ranking order as the primary basis for evaluation. Next, we consider rewards for high scores for high-ranked gestures and low scores for low-ranked gestures. %We refer this as the relevance.
We also quantify correspondence between the trends of output and ground truth scores. Finally, we account for disentanglement between identity features of gestures as a discounting factor. 
Finally, these are combined using appropriate weights resulting in what is referred to as the \textbf{\texttt{advanced acceptance score}} ($A_r^*$). 
This can serve as a standalone measure for holistic evaluation. To assess the effectiveness of the proposed \textbf{\texttt{advanced acceptance score}}, we perform extensive experimentation over three datasets and five state-of-the-art (SOTA) models. Results show that the optimal score selected with our measure is more holistic than other existing measures. Furthermore, we conduct exhaustive ablation studies to demonstrate the reliability and sensitivity of this score to individual measures based on the chosen weights.

We propose four measures and combine them with adequate weights in the advanced acceptance score.
1. Rank deviation ($\hat{r}$)
2. Relevance ($\mathcal{R}$)
3. Trend match distance ($\Psi$)
4. ICGD Score ($C_d$)


## Requirements

1. numpy
2. sckit-learn
3. scipy
4. tensorflow $\geq$ 2.8.0
5. biomQuant

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


