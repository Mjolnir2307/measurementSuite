####### Importing Libraries
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from src.DGBQA_Score import gbqa_delta_dist_compute
from src.ICGDScore import CGID_Score_Calculator
from src.RankDeviation import avg_rank_deviation
from src.AcceptanceScore import acceptance_score
from src.PatternMatchDistance import pattern_match_dist

##### Helper functions
def get_val(embeddings, 
        y_dev, 
        y_dev_id, 
        eer_values, 
        G_total,
        I_total,
        alpha,
        beta,
        lambda_scale,
        kappa,
        nu):
    
    """
    Function to return evaluation measures: [r,R,d,C_d,nAr]
    """

    dgbqa_score = [] # DGBQA Score

    for g_id in range(G_total):
        dgbqa_score_curr, _, _, _ = gbqa_delta_dist_compute(embeddings,g_id,I_total,y_dev,y_dev_id)
        dgbqa_score.append(dgbqa_score_curr)

    dgbqa_score = np.array(dgbqa_score) # Array Formation
    dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
    dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

    e_prime = 100 - np.array(eer_values)
    e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
    e_prime = e_prime/np.linalg.norm(e_prime)

    r = avg_rank_deviation(eer_values,dgbqa_score,G_total)
    Ar = acceptance_score(dgbqa_score,
                      e_prime,
                      G_total,
                      normalizer=False,
                      relevance=False,
                      lambda_scale=lambda_scale,
                      kappa=kappa)
    R = acceptance_score(dgbqa_score,
                                e_prime,
                                G_total,
                                normalizer=False,
                                relevance=True,
                                lambda_scale=lambda_scale,
                                kappa=kappa)
    d = pattern_match_dist(dgbqa_score,e_prime,G_total)
    d_metric = (np.log2(2+nu*d)**(-1/alpha))
    C_I, C_D = CGID_Score_Calculator(embeddings,y_dev)
    O_prime = np.exp(-beta*C_D) # Orthogonal Penalty
    Ar_star = Ar*d_metric*O_prime
    Ar_max = acceptance_score(dgbqa_score,
                            e_prime,
                            G_total,
                            normalizer=True,
                            relevance=False,
                            lambda_scale=lambda_scale,
                            kappa=kappa)
    nAr_star = Ar_star/Ar_max
    
    values = np.array([r,R,d,C_D,nAr_star])
    return values

def get_measures(embedding_list, 
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
    
    """Function to get measures corresponding to the models in the embedding list"""

    y_dev = np.load(y_dev_path)['arr_0']
    y_dev_id = np.load(y_dev_id_path)['arr_0']
    measure_val = []

    for embedding_path in embedding_list:
        embedding = np.load(embedding_path)['arr_0']
        measure_val.append(get_val(embedding,
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
    return np.array(measure_val)

###### Radar scale
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta