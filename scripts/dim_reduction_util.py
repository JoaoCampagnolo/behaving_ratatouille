# Import packages:
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import time
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
import hdbscan
import seaborn as sns
import sklearn.cluster as cluster
from util import get_time



def jhc_tsne(data, perplexity=32, relTol=1e-4, num_tsne_dim=2, sigmaTolerance=1e-5, momentum=.5, final_momentum=0.8,
             mom_switch_iter=250, stop_lying_iter=125, lie_multiplier=4, max_iter=1000, epsilon=500, min_gain=.01,
             tsne_readout=1, embedding_batchSize=20000, maxOptimIter=100, trainingSetSize=35000, kdNeighbors=5,
             training_relTol=2e-3, training_perplexity=20, training_numPoints=10000, minTemplateLength=1):
    '''
    Yield training set: partition the dataset
    Created by G. Berman and modified by João Campagnolo.
    '''
        
    tsne_pars = {'perplexity':perplexity, 'relTol':relTol, 'num_tsne_dim':num_tsne_dim, 'sigmaTolerance':sigmaTolerance,
                 'maxNeighbors':maxNeighbors, 'momentum':momentum, 'final_momentum':final_momentum, 'mom_switch_iter':mom_switch_iter,
                 'stop_lying_iter':stop_lying_iter, 'lie_multiplier':lie_multiplier, 'max_iter':max_iter, 'min_gain':min_gain,
                 'tsne_readout':tsne_readout, 'embedding_batchSize':embedding_batchSize, 'maxOptimIter':maxOptimIter, 
                 'trainingSetSize':trainingSetSize, 'kdNeighbors':kdNeighbors, 'training_relTol':training_relTol, 
                 'training_perplexity':training_perplexity, 'training_numPoints':training_numPoints, 
                 'minTemplateLength':minTemplateLength
                }
  
    # Find the KL divergence amongst the columns of the expression matrix (data)
    N = np.shape(data)[1]
    dist_matx = np.zeros((N,N), dtype=float)
    print(f'Creating {N}x{N} pair-wise distance matrix with KL diverge')
    for j in range(N): #cols
        for i in range(N): #rows
            dist_matx[i][j] = kld(data[:,i],data[:,j])
    dist_matx /= max(dist_matx)
    
    # D2P Identifies appropriate sigma's to get kk NNs up to some tolerance 
    # see https://github.com/gordonberman/MotionMapper/blob/master/t_sne/d2p_sparse.m
    P, betas = _d2p_sparse_(D**2, u=perplexity, tol=sigmaTolerance)

    return dist_matx #FIXME


def _get_distance_matrix_(X, metric='euclidean'):
    '''
    Compute the distance matrix from a vector array X.
Valid values for metric are:
    *From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs.
    *From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’].
    *Precomputed (e.g. Kullback-Leibler Divergence, kld)
    '''
    time_start = time.time()
    dist_matx = pairwise_distances(X)
    print('Distance matrix computed. Time elapsed: {} seconds'.format(time.time()-time_start))
    return dist_matx

def _apply_PCA_(train_data, num_components=70):
    '''
    Fit PCA to reduce dimension. This might mattern in case GMM is used. For t-SNE, it serves no purpose since t-SNE 
conserves local structure, as oposed to PCA, which conserves global structure.
    '''
    print("Applying pca")
    pca = PCA(n_components=num_components)
    pca_data = pca.fit_transform(train_data)
    print(f"PCA data shape: {np.shape(pca_data)}")
    print("Total explained variance:", np.cumsum(pca.explained_variance_ratio_)[num_components-1])
    return pca_data

def apply_tsne(dist_matx, per=50, ee=20, lr=100, n=3500, met='precomputed'):
    '''
    Fit t-SNE to reduce dimension. 
    NB. With t-SNE, it might be of interest to check the fitting for different parameters and metrics since the performance varies a lot from dataset to dataset.
    The accepted metrics are those accepted by sklearn.pairwise_distances, or precomputed metrics/divergences... (such as KLD). KLD is suited to assess distances between probability distributions, unlike the Euclidian metric.
    '''
    time_start = time.time()
    tsne = TSNE(n_components=2, perplexity=per, early_exaggeration=ee,
                          learning_rate=lr, n_iter=n, n_iter_without_progress=300, 
                          min_grad_norm=1e-07, metric=met, init='random', verbose=2, 
                          random_state=None, method='barnes_hut', angle=0.3)
    tsne_results = tsne.fit_transform(dist_matx)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print('t-SNE train data shape:{}'.format(np.shape(tsne_results)))
    return tsne_results

def kld(p, q):
    '''
    Kullback-Leibler divergence D(P||Q) for discrete distributions
    Parameters:
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    '''
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    # Add error message when p==0
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def _d2p_sparse_(D, maxNeighbors=150, u=15, tol=10^-4):
    '''
    Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
    kernel with a certain uncertainty for every datapoint. The desired
    uncertainty can be specified through the perplexity u (default = 15). The
    desired perplexity is obtained up to some tolerance that can be specified
    by tol (default = 1e-4).
    The function returns the final Gaussian kernel in P, as well as the 
    employed precisions per instance in beta.
    
    Input variables:

        D -> NxN distance matrix 
        u -> perplexity. 2^H (H is the transition entropy)
        tol -> binary search tolerance for finding pointwise transition
               region
        maxNeighbors -> maximum number of non-zero neighbors in P


    Output variables:

        beta -> list of individual area parameters
        P -> sparse transition matrix


    (C) Laurens van der Maaten, 2008
    Maastricht University

    Adapted to Python by João Campagnolo, 2022
    '''
    n = np.shape(D)[1]               
    if maxNeighbors >= n:
        maxNeighbors = n-1
    
    beta = np.ones(n) 
    logU = np.log(u)  
    
    jj = np.zeros((n,maxNeighbors))
    vals = np.zeros_like(jj)
    
    # Run over all datapoints
    for i in range(n):
        
        if i%500==0:
            print(f'Computed P-values: {i} of {n} datapoints')
        
        # Set minimum and maximum values for precision
        betamin = -float('inf')
        betamax = float('inf')
        
        q = D[:,i]
        sortVals = np.sort(q)
        sortIdx = np.argsort(q)
        sortVals = sortVals[1:maxNeighbors]
        sortIdx = sortIdx[1:maxNeighbors]
        jj[i,:] = sortIdx
        
        # Compute the Gaussian kernel and entropy for the current precision
        H, thisP = _Hbeta_(sortVals, beta[i])
        
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if isinf(betamax):
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
                    
            else:
                betamax = beta[i]
                if isinf(betamin):
                    beta(i) = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            
            # Recompute the values
            H, thisP = _Hbeta_(sortVals, beta[i])
            Hdiff = H - logU
            tries = tries + 1        

        vals[i,:] = thisP

    ii = np.asarray([range(n)]*maxNeighbors).reshape(n*maxNeighbors,1)
    jj = jj.reshape(n*maxNeighbors,1)
    vals = vals.reshape(n*maxNeighbors,1)
    
    new_b = 1. / np.sqrt(beta)
    print(f'Mean value of sigma: {np.mean(new_b)}')
    print(f'Minimum value of sigma: {min(new_b)}')
    print(f'Maximum value of sigma: {max(new_b)}')
    
    return
    
def _Hbeta_(D, beta):
    '''
    Function that computes the Gaussian kernel values given a vector of
    squared Euclidean distances, and the precision of the Gaussian kernel.
    The function also computes the perplexity of the distribution.
    '''
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D,P)) / sumP;
    P /= sumP
    return H, P