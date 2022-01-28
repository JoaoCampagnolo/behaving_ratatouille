# File:             Segmentation.py
# Date:             March 2019
# Description:      Having embedded the data into a 2-dimension space, the user is faced with discretization.
#                   This will be acheived through the creation of a grid, which will be sustain the positions of
#                   data points and where a segmentation technique will be employed.
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages:
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
from segmentation_tools import *



class _Clust_alg(Enum):
    Watershed=0
    GMM=1
    HDBSCAN=2
    HAC=3
    random=4
    
    
class Segmentation():
    def __init__(self, data, technique=0, mesh_mag=1, xmax=600, ymax=600, mesh_margins=10, gk_order=0, gk_mode='nearest',
                 gk_trunk=4.0, gk_size=6, wat_mdist=5, wat_trsh=0.001, wat_dmask=0, wat_tmask=0, use_hd_slk=False, hd_minsz=30,
                 hd_minsamp=15, hd_metric='euclidean', hd_alpha=1.0, hd_leafsz=50, hd_select='eom', slt_cut=6, hdbscan_min=5,
                 n_clust=125, gmm_covtype='full', hac_aff='euclidean', hac_con=None, hac_tree='auto', hac_link='ward', 
                 hac_dist=None, hac_compdst=False):
        
        self.data = data
        self.technique = technique
        
        # Parameter dictionary
        self.params = {'mesh': {'magnitude':mesh_mag, 'xmax':xmax, 'ymax':ymax, 'margin':mesh_margins},
                       
                       '2D_Gauss_Conv': {'function': {'order':gk_order, 'output':None, 'mode':gk_mode,
                                                      'cval':0.0, 'truncate':gk_trunk},
                                         'kernel': {'kernel_size':gk_size, 'gamma':0.4}},
                       
                       'Watershed': {'local_maxima': {'min_distance':wat_mdist, 'threshold_rel':wat_trsh , 'indices':False}, 
                                     'mask': {'min_distance':wat_dmask, 'threshold_rel':wat_tmask , 'indices':False}},
                       
                       'HDBSCAN': {'settings:':{'min_cluster_size':hd_minsz, 'min_samples':hd_minsamp, 'metric':hd_metric,
                                   'alpha':hd_alpha, 'leaf_size':hd_leafsz, 'cluster_selection_method':hd_select, 
                                   'p':None},
                                   'single_linkage_tree':{'cut_distance':slt_cut, 'min_cluster_size':hd_minsz}},
                                              
                       'GMM': {'n_components':n_clust, 'covariance_type':gmm_covtype},
                       
                       'HAC': {'n_clusters':n_clust, 'affinity':hac_aff, 'connectivity':hac_con, 
                               'compute_full_tree':hac_tree, 'linkage':hac_link, 'distance_threshold':hac_dist,
                               'compute_distances':hac_compdst}
                      }
        
        self.time_start = time.time()
        
        # Clustering dictionary
        self.clustering_dict = {}
        
        # Wateshed
        if self.technique == _Clust_alg.Watershed.value:
            print(f'Performing Watershed clustering with parameters: {self.params["Watershed"]}')
            # Build mesh
            self.data *= self.params['mesh']['magnitude']
            self.data[:,0]+=self.params['mesh']['xmax']/2
            self.data[:,1]+=self.params['mesh']['ymax']/2
            self.data = np.round(self.data).astype(np.int)
            self.xmax, self.ymax = self.params['mesh']['xmax'], self.params['mesh']['ymax']
            self.mesh, self.pix_to_point_idx, self.point_idx_to_pix = self.map_mesh(self.data, self.xmax, self.ymax)
            # Apply 2D Gaussian convolution
            self.kernel_size = self.params['2D_Gauss_Conv']['kernel']['kernel_size']
            self.prob_dens_f = gaussian_filter(self.mesh, self.kernel_size, **self.params['2D_Gauss_Conv']['function'])
            self.prob_dens_f = exposure.adjust_gamma(self.prob_dens_f, self.params['2D_Gauss_Conv']['kernel']['gamma'])
            self.prob_dens_f = self.prob_dens_f / self.prob_dens_f.sum()            
            # Apply Watershed
            self.distances = ndi.distance_transform_edt(self.prob_dens_f)
            self.loc_maxima = peak_local_max(self.prob_dens_f, **self.params['Watershed']['local_maxima']) #
            self.mask = peak_local_max(self.prob_dens_f, **self.params['Watershed']['mask'])
            self.markers, self.num_class = ndi.label(self.loc_maxima)
            self.mesh_labels = watershed(-self.prob_dens_f, self.markers, mask=self.mask)
            self.clust_labels = self.get_mesh_labels(self.mesh_labels, self.data)
            self.num_clust = np.ndarray.max(self.clust_labels)
            print(f'Number of clusters: {self.num_clust}')
            self.clustering_dict['Clust_technique'] = 'Watershed'
            self.clustering_dict['Parameters'] = {'Mesh':self.params['mesh'], 'Gauss_conv':self.params['2D_Gauss_Conv'],
                                                  'Water_clust':self.params['Watershed']}
            self.clustering_dict['Data'] = {'PDF':self.prob_dens_f, 'Pixel_tpoint':self.pix_to_point_idx,
                                            'Point_tpixel':self.point_idx_to_pix, 'Mesh_labels':self.mesh_labels,
                                            'Frame_cluster':self.clust_labels}            
            
        # Gaussian Mixture Models
        if self.technique == _Clust_alg.GMM.value:
            print(f'Performing GMM clustering with parameters: {self.params["GMM"]}')      
            self.classifier = mixture.GaussianMixture(**self.params['GMM'])
            self.model_fit = self.classifier.fit(self.data)
            self.clust_labels = self.classifier.predict(self.data)
            self.num_clust = max(self.clust_labels)+1
            self.post_proba = self.classifier.predict_proba(self.data)
            self.score = self.classifier.score(self.data)
            self.bic = self.classifier.bic(self.data)
            self.aic = self.classifier.aic(self.data)
            print(f'Number of clusters: {self.num_clust}')
            self.clustering_dict['Clust_technique'] = 'GMM'
            self.clustering_dict['Parameters'] = self.params['GMM']
            self.clustering_dict['Data'] = {'Frame_cluster':self.clust_labels, 'AIC':self.aic, 'BIC':self.bic}
            
        # HDBSCAN
        if self.technique == _Clust_alg.HDBSCAN.value:
            print(f'Performing HDBSCAN clustering with parameters: {self.params["HDBSCAN"]}')    
            self.classifier = hdbscan.HDBSCAN(**self.params['HDBSCAN'])
            self.model_fit = self.classifier.fit(self.data)
            self.post_proba = self.classifier.probabilities_
            self.clust_labels = self.model_fit.labels_
            self.single_linkage_tree = []
            if use_hd_slk:
                self.clust_labels = self.classifier.single_linkage_tree_.get_clusters(**self.params['single_linkage_tree'])
                self.single_linkage_tree = self.model_fit.single_linkage_tree_#.plot()
                #.plot(select_clusters=True,selection_palette=sns.color_palette('deep', 8))
            #self.min_span_tree = self.model_fit.minimum_spanning_tree_#.plot()
            self.cond_tree = self.model_fit.condensed_tree_#.plot()
            self.hdbscan_scores = self.model_fit.outlier_scores_   
            self.num_clust = max(self.clust_labels)+1
            print(f'Number of clusters: {self.num_clust}')
            self.clustering_dict['Clust_technique'] = 'HDBSCAN'
            self.clustering_dict['Parameters'] = self.params['HDBSCAN']
            self.clustering_dict['Data'] = {'Frame_cluster':self.clust_labels, 'Condensed_tree':self.cond_tree, 
                                            'Single_link_tree':self.single_linkage_tree}            
            
            
        # Hierarchical Aglomerative Clustering
        if self.technique == _Clust_alg.HAC.value:
            print(f'Performing HAC clustering with parameters: {self.params["HAC"]}')
            self.classifier = AgglomerativeClustering(**self.params['HAC'])
            self.model_fit = classifier.fit(self.data)
            self.clust_labels = model_fit.labels_
            self.num_clust = max(self.clust_labels)+1
            print(f'Number of clusters: {self.num_clust}')
            self.clustering_dict['Clust_technique'] = 'HAC'
            self.clustering_dict['Parameters'] = self.params['HAC']
            self.clustering_dict['Data'] = {'Frame_cluster':self.clust_labels}
            # Check how to plot hierarchy later: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
            
        # Random
        if self.technique == _Clust_alg.random.value:
            print(f'Assigning {n_clust} random labels')
            self.clust_labels = np.asarray([random.randint(0,gmm_comps-1) for _ in range(np.shape(low_dim_data)[0])])
            
        # Elapsed time
        self.dt = time.time()-self.time_start
        print(f'Clustering completed. Time elapsed: {self.dt} seconds')            
            
            
          
        
        
        
        
        
        
        # Functions to use:
    def map_mesh(self, low_dim_data, xmax, ymax):
        # frame the x,y coordinates from the low dimensional data.
        self.mesh = np.zeros(shape=(xmax, ymax))
        self.point_idx_to_pix = dict()
        self.pix_to_point_idx = dict()
        for x in range(xmax):
            for y in range(ymax):
                self.pix_to_point_idx[(x,y)] = list()
        for idx, p in enumerate(low_dim_data):
            x, y = p[0], p[1]
            if x<0 or x>xmax or y<0 or y>ymax:
                continue
            self.mesh[x, y] += 1
            self.pix_to_point_idx[(x,y)].append(idx) #each entry(x,y) in this dict is the index of the frame
            self.point_idx_to_pix[idx] = (x,y) #each entry(frame) in this dict is the postition of the pixel
        return self.mesh, self.pix_to_point_idx, self.point_idx_to_pix
    
    def get_mesh_labels(self, mesh, data):
        labels = np.zeros(shape=np.shape(data)[0])
        for y in range(np.shape(mesh)[0]):
            for x in range(np.shape(mesh)[1]):
                frames = self.pix_to_point_idx[(x,y)]
                for i in range(len(frames)):
                    labels[frames[i]] = self.clust_labels[x,y]
        return labels
                
    
    
    