# File:             segmantation_tools.py
# Date:             Winter 2022
# Description:      Some utils that are usefull in the segmentation stage. 
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages
import numpy as np
import time
from util import get_time
from enum import Enum
from scipy.ndimage.filters import gaussian_filter
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from skimage import exposure
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.filters.rank import gradient
import random
import hdbscan

def map_mesh(low_dim_data, xmax, ymax):
    # frame the x,y coordinates from the low dimensional data.
    mesh = np.zeros(shape=(xmax, ymax))
    point_idx_to_pix = dict()
    pix_to_point_idx = dict()
    for x in range(xmax):
        for y in range(ymax):
            pix_to_point_idx[(x,y)] = list()
    for idx, p in enumerate(low_dim_data):
        x, y = p[0], p[1]
        if x<0 or x>xmax or y<0 or y>ymax:
            continue
        mesh[x, y] += 1
        pix_to_point_idx[(x,y)].append(idx) #each entry(x,y) in this dict is the index of the frame
        point_idx_to_pix[idx] = (x,y) #each entry(frame) in this dict is the postition of the pixel
    return mesh, pix_to_point_idx, point_idx_to_pix

def get_mesh_labels(mesh, data):
    labels = np.zeros(shape=np.shape(data)[0])
    for y in range(np.shape(mesh)[0]):
        for x in range(np.shape(mesh)[1]):
            frames = pix_to_point_idx[(x,y)]
            for i in range(len(frames)):
                labels[frames[i]] = clust_labels[x,y]
    return labels
    
def build_mesh(data, magnitude, xmax, ymax):
    data *= magnitude
    data[:,0]+=xmax/2
    data[:,1]+=ymax/2
    data = np.round(data).astype(np.int)
    mesh, pix_to_point_idx, point_idx_to_pix = map_mesh(data, xmax, ymax)
    return mesh, pix_to_point_idx, point_idx_to_pix

def gaussian_convol(mesh, kernel_size, function, gamma):
    prob_dens_f = gaussian_filter(mesh, kernel_size, **function)
    prob_dens_f = exposure.adjust_gamma(prob_dens_f, gamma)
    prob_dens_f = prob_dens_f / prob_dens_f.sum()   
    return prob_dens_f

def do_watershed(data, prob_dens_f, local_maxima, mask_pars):
    distances = ndi.distance_transform_edt(prob_dens_f)
    loc_maxima = peak_local_max(prob_dens_f, **local_maxima) #
    mask = peak_local_max(prob_dens_f, **mask_pars)
    markers, num_class = ndi.label(loc_maxima)
    mesh_labels = watershed(-prob_dens_f, markers, mask=mask)
    clust_labels = get_mesh_labels(mesh_labels, data)
    num_clust = np.ndarray.max(clust_labels)
    return clust_labels

def do_gmms(data, gmm_pars):
    classifier = mixture.GaussianMixture(**gmm_pars)
    model_fit = classifier.fit(data)
    clust_labels = classifier.predict(data)
    post_proba = classifier.predict_proba(data)
    score = classifier.score(data)
    bic = classifier.bic(data)
    aic = classifier.aic(data)
    return clust_labels, post_proba, score, bic, aic

def do_hdbscan(data, pars, use_hd_slk=False, slk_pars=None):
    classifier = hdbscan.HDBSCAN(**pars)
    model_fit = classifier.fit(data)
    post_proba = classifier.probabilities_
    clust_labels = model_fit.labels_
    single_linkage_tree = []
    if use_hd_slk:
        clust_labels = classifier.single_linkage_tree_.get_clusters(**slk_pars)
        single_linkage_tree = model_fit.single_linkage_tree_#.plot()
        #.plot(select_clusters=True,selection_palette=sns.color_palette('deep', 8))
    #min_span_tree = model_fit.minimum_spanning_tree_#.plot()
    cond_tree = model_fit.condensed_tree_#.plot()
    hdbscan_scores = model_fit.outlier_scores_ 
    return clust_labels, single_linkage_tree, cond_tree, hdbscan_scores

def do_hac(data, hac_pars):
    classifier = AgglomerativeClustering(**hac_pars)
    model_fit = classifier.fit(data)
    clust_labels = model_fit.labels_
    return clust_labels