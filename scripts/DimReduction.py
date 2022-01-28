# File:             DimReduction.py
# Date:             March 2019
# Description:      Allows the user to lower the dimensioanlity of the dataset, while minimizing the loss of information 
#                   in doing so. Here, the user is free to employ a few techniques such as PCA (preserves +global structure),
#                   t-SNE (preserves +local structure), Umap, etc., or even to add newer techniques. The embedded data points
#                   can later be subjected to segmentation techniques to acheive discrete representations of behavior.
# Author:           Joao Campagnolo
# Python version:   Python 3.7+

# TODO: Improve function description

# Import packages:
import numpy as np
import time
from util import get_time
from dim_reduction_util import kld
from scipy.stats import entropy
from enum import Enum
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from parametric_tsne import ParametricTSNE
from par_tsne_core import Parametric_tSNE
from sklearn.manifold import Isomap
import umap
from segmentation_tools import *
# from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP


class _Dim_red_alg(Enum): #move to the end
    kernel_PCA=0
    t_SNE=1
    Par_tsne=2 #https://github.com/luke0201/parametric-tsne-keras 
    Par_tsne2=3 #https://github.com/jsilter/parametric_tsne
    Isomap=4
    Umap=5
    Par_Umap=6
    PCA=7

__linear = [False, False, False, False, False, False, True] # is it linear dimensionality reduction technique
__non_linear = [not elem for elem in __linear]

class DimReduction():
    
    def __init__(self, train_data, test_data, technique, embed_test=False, components=2, kpca_kernel='sigmoid',
                 kpca_gamma=None, kpca_kernel_degre=3, solver='auto', n_iter=1000, random_state=None, tsne_perplexity=30,
                 n_iter_without_progress=100, init='random', learning_rate=200.0, metric='euclidean', verbose=0, n_jobs=1,
                 min_grad_norm=1e-07, method='barnes_hut', angle=0.5, early_exaggeration=12.0, ismp_nn=5, ismp_toler=0,
                 ismp_path='auto', ismp_nb_alg='auto', umap_rd_st=42, umap_nn=15, umap_min_dst=0.1, umap_epochs=None, d0=10,
                 d1=100, d2=2, pumap_save_embed=False, save_embed_path=None, pumap_load_embed=False, load_embed_path=None,
                 par_tsne_alpha=1.0, par_tsne_opt='adam', part_tsne_batch_sz=64, par_tsne_seed=0, make_trainset=False,
                 square_distances=True):
        
        self.train_data = train_data
        self.test_data = test_data
        self.technique = technique
        self.make_trainset = make_trainset
        self.embed_algs = ['kernel_PCA', 't_SNE', 'Par_tsne', 'Par_tsne2', 'Isomap', 'Umap', 'Par_Umap', 'PCA']
        
        # Parameter dictionary
        self.params = {'kernel_PCA': {'n_components':components, 'kernel':kpca_kernel, 'gamma':kpca_gamma,
                                      'degree':kpca_kernel_degre, 'eigen_solver':solver, 'max_iter':n_iter, 
                                      'random_state': random_state}, #use int random state for reproducible results
                       
                       't_SNE': {'n_components':components, 'perplexity':tsne_perplexity,
                                 'early_exaggeration':early_exaggeration, 'learning_rate':learning_rate, 'n_iter':n_iter, 
                                 'n_iter_without_progress':n_iter_without_progress, 'n_jobs':n_jobs,
                                 'min_grad_norm': min_grad_norm, 'metric':metric, #or euclidean or 'precomputed' or 'cosine'
                                 'init':init, 'verbose':verbose, 'random_state':random_state, 'method':method, 'angle':angle,
                                 'square_distances':square_distances},
                       
                       'Par_tsne':{'n_components':components, 'perplexity':tsne_perplexity, 
                                   'n_iter':n_iter, 'verbose':verbose},
                       
                       'Par_tsne2':{'alpha':par_tsne_alpha, 'optimizer':par_tsne_opt, 'batch_size':part_tsne_batch_sz,
                                    'seed':par_tsne_seed},
                       
                       'Isomap': {'n_neighbors':ismp_nn, 'n_components':components, 'eigen_solver':solver, 'tol':ismp_toler,
                                  'max_iter':n_iter, 'path_method':ismp_path, 'neighbors_algorithm':ismp_nb_alg,
                                  'metric':metric}, #try 'minkowski'
                       
                       'Umap': {'random_state':umap_rd_st, 'n_neighbors': umap_nn, 'min_dist': umap_min_dst, 
                                'n_components':components, 'metric': metric, 'n_epochs':umap_epochs},
                       
                       'Par_Umap': {'Network':{'dims':(d0,d1,d2), 'n_components':components},
                                    'Embedder':{'save_embed':save_embed_path, 'save_path':save_embed_path,
                                                'load_embed':pumap_load_embed, 'load_path':load_embed_path}},
                       
                       'PCA': {'n_components': components},
                      }
                
        self.time_start = time.time()
        
        # Embedding dictionary
        self.embedding_dict = {} 
        
        # Create a training set for parametric embeddings: 
        # Approach: divide and conquer - split the fully preprocessed datasets into tranches and perform 
        # a preliminary embedding of tranches of the data. Then yield a representative set of points from
        # each tranche. Combine the sets from all tranches to get the complete training set. Embed the 
        # the training set with a parametric embedding technique. Finally, embed the remaining dataset
        # with the parametrized embedder.
        #if self.make_trainset:
             
        
        # Kernel PCA
        if self.technique == _Dim_red_alg.kernel_PCA.value:
            print(f'Performing kernel PCA with parameters: {self.params["kernel_PCA"]}')
            kpca = KernelPCA(**self.params['kernel_PCA'])
            self.data_embed = kpca.fit_transform(np.transpose(self.train_data))
            self.embedding_dict['Embed_technique'] = 'kernel_PCA'
            self.embedding_dict['Parameters'] = self.params["kernel_PCA"]
            self.embedding_dict['Data'] = {'Embed_data':self.data_embed}
        
        # t-SNE
        if self.technique == _Dim_red_alg.t_SNE.value:
            print(f'Performing t-SNE with parameters: {self.params["t_SNE"]}')
            if metric == 'kld':
                self.params['t_SNE']['metric'] = kld
            if metric == 'entropy':
                self.params['t_SNE']['metric'] = entropy            
            tsne = TSNE(**self.params['t_SNE'])
            self.data_embed = tsne.fit_transform(np.transpose(self.train_data))
            self.embedding_dict['Embed_technique'] = 't_SNE'
            self.embedding_dict['Parameters'] = self.params["t_SNE"]
            self.embedding_dict['Data'] = {'Embed_data':self.data_embed}
            
        # Parametric t-SNE #1 (Par_tsne)
        if self.technique == _Dim_red_alg.Par_tsne.value:
            print(f'Performing t-SNE with parameters: {self.params["Par_tsne"]}')
            embedder = ParametricTSNE(**self.params['Par_tsne'])
            self.data_embed = embedder.fit_transform(np.transpose(self.train_data))
            self.embedding_dict['Embed_technique'] = 'Par_tsne'
            self.embedding_dict['Parameters'] = self.params["Par_tsne"]
            self.embedding_dict['Data']['Embed_data'] = self.data_embed
            if embed_test:
                self.test_embed = embedder.transform(np.transpose(self.test_data))
                self.embedding_dict['Data']['Embed_test_data'] = self.test_data
                
        # Parametric t-SNE #2 (Par_tsne2)
        if self.technique == _Dim_red_alg.Par_tsne2.value:
            print(f'Performing t-SNE with parameters: {self.params["Par_tsne2"]}')
            high_dim = np.shape(self.train_data)[0]
            nn_layers = [layers.Dense(d0, input_shape=(high_dim,), activation='sigmoid', kernel_initializer='glorot_uniform'),
                      layers.Dense(d1, activation='sigmoid', kernel_initializer='glorot_uniform'),
                      layers.Dense(d2, activation='relu', kernel_initializer='glorot_uniform')]
            embedder = Parametric_tSNE(high_dim, components, tsne_perplexity, 
                                       all_layers=nn_layers, **self.params['Par_tsne2'])
            embedder.fit(np.transpose(self.train_data))
            self.data_embed = embedder.transform(np.transpose(self.train_data))
            self.embedding_dict['Embed_technique'] = 'Par_tsne2'
            self.embedding_dict['Parameters'] = self.params["Par_tsne2"]
            self.embedding_dict['Data'] = {'Embed_data':self.data_embed, 'Embedder':embedder}            
            if embed_test:
                self.test_embed = embedder.transform(np.transpose(self.test_data))
                self.embedding_dict['Data']['Embed_test_data'] = self.test_data
            
        # Isometric mapping (Isomap)
        if self.technique == _Dim_red_alg.Isomap.value:
            print(f'Performing Isomap with parameters: {self.params["Isomap"]}')
            isomap = Isomap(**self.params['Isomap'])
            self.data_embed = isomap.fit_transform(np.transpose(self.train_data))
            self.embedding_dict['Embed_technique'] = 'Isomap'
            self.embedding_dict['Parameters'] = self.params["Isomap"]
            self.embedding_dict['Data']['Embed_data'] = self.data_embed            

        # Uniform Manifold mapping (Umap)
        if self.technique == _Dim_red_alg.Umap.value:
            print(f'Performing Umap with parameters: {self.params["Umap"]}')
            reducer = umap.UMAP(**self.params['Umap'])
            self.data_embed = reducer.fit_transform(np.transpose(self.train_data))
            self.embedding_dict['Embed_technique'] = 'Umap'
            self.embedding_dict['Parameters'] = self.params["Umap"]
            self.embedding_dict['Data']['Embed_data'] = self.data_embed             
        
        # Parametric Uniform Manifold mapping (Par_Umap): this is the more complex model so far. 
        # Key advantage: ability to save and load an encoder.
        # UMAP IS NOT FIT FOR PUBLIC USE YET
#         if self.technique == _Dim_red_alg.Par_Umap.value:
#             print(f'Performing parametric Umap with parameters: {self.params["Par_Umap"]}')
#             # Defining an encoder - first, check for pre-existing encoders
#             if self.params['Par_Umap']['Embedder']['load_embed']:
#                 if self.params['Par_Umap']['Embedder']['load_path'] is not None:
#                     embedder = load_ParametricUMAP(self.params['Par_Umap']['Embedder']['load_path'])
#                 else:
#                     print('Failed to load Par_Umap embedder. Please provide path: (load_embed_path=__path__)')
#                     return
#             else:
#                 dims = self.params['Par_Umap']['Encoder']['dims']
#                 encoder = _create_encoder(dims, components)
#                 encoder.summary()
#                 embedder = ParametricUMAP(encoder=encoder, dims=dims)
#                 if self.params['Par_Umap']['Embedder']['save_embed']:
#                     if self.params['Par_Umap']['Embedder']['save_path'] is not None:
#                         embedder.save(self.params['Par_Umap']['Embedder']['save_path'])
#                     else:
#                         print('Failed to save Par_Umap embedder. Please provide path: (save_embed_path=__path__)')
#                         return
#             self.data_embed = embedder.fit_transform(self.train_data)
            
        # Principal Component Analysis
        if self.technique == _Dim_red_alg.PCA.value:
            print(f'Performing PCA with parameters: {self.params["PCA"]}')
            pca = PCA(**self.params['PCA'])
            pca.fit(np.transpose(self.train_data))   
            self.exp_var_ratio = pca.explained_variance_ratio_
            self.data_embed = pca.fit_transform(np.transpose(self.train_data))
            self.embedding_dict['Embed_technique'] = 'PCA'
            self.embedding_dict['Parameters'] = self.params["PCA"]
            self.embedding_dict['Data'] = {'Embed_data':self.data_embed, 'Exp_var_ratio':self.exp_var_ratio}
            
        # Get embedded data shape
        print(f'Embedded data shape: {np.shape(self.data_embed)}')
        if embed_test:
            print(f'Embedded test data shape: {np.shape(self.test_embed)}')
        
        # Elapsed time
        self.dt = time.time()-self.time_start
        print(f'Dimensionality reduction completed. Time elapsed: {self.dt} seconds')
        

              
            
            
            
            
            
            
            
            
       # Functions to use
    def _create_encoder(dims, n_components):
        encoder
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=dims),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), 
                                   activation="relu", padding="same"
                                  ),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu", 
                                   padding="same"
                                  ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=256, activation="relu"),
            tf.keras.layers.Dense(units=256, activation="relu"),
            tf.keras.layers.Dense(units=n_components),
        ])
        encoder.summary()
        return encoder
                         
    def _fit_tSNE(data, **kwards):
        tsne = TSNE(**kwards)
        parameters = tsne.get_params(deep=True)
        print(f't-SNE parameters: {parameters}')
        data_embed = tsne.fit_transform(data)
        return data_embed, parameters
    
    def _get_distance_matrix(X, metric):
        '''
        Compute the distance matrix from a vector array X.
    Valid values for metric are:
        *From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs.
        *From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’].
        *Precomputed (e.g. Kullback-Leibler Divergence, 'kl_divergence' - slower)
        '''
        time_start = time.time()
        dist_matx = pairwise_distances(X, metric)
        print('Distance matrix computed. Time elapsed: {} seconds'.format(time.time()-time_start))
        return dist_matx

    def _kl_divergence(p, q):
        """Kullback-Leibler divergence D(P||Q) for discrete distributions
        Parameters:
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
        """
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)

        # Add error message when p==0
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))




