
from time import perf_counter_ns
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

from annindex.pq import ProductQuantization

def get_cluster_match(found_clusters, true_clusters):
    cluster_dist = euclidean_distances(found_clusters, true_clusters)
    row_ind, col_ind = linear_sum_assignment(cluster_dist)
    match_dist = cluster_dist[row_ind,col_ind]
    return row_ind, col_ind, match_dist

def iterate_subvectors(pq, *arrays):
    """
    Generate tuples of subvector arrays
    """
    for i in range(pq.n_chunks):        
        s = i * pq.chunk_dim
        e = (i+1) * pq.chunk_dim
        
        yield (pq.kms[i].cluster_centers_, ) + tuple( X[:, s:e] for X in arrays )
        

def show_clustering(X, true_centers, pq):
    """
    Show clustering using the first 2 dimensions from each chunk.
    """
    for chunk, (found, data, truth) in enumerate(iterate_subvectors(pq, X, true_centers)):
        plt.figure()
        plt.plot(data[:,0], data[:,1], 'go', label='data', alpha=0.1, mec='none')
        plt.plot(truth[:,0], truth[:,1], 'ko', label='ground truth', mfc='none')
        plt.plot(found[:,0], found[:,1], 'rx', label='k-means')
        
        row_ind, col_ind, match_dist = get_cluster_match(found, truth)        
        for i, j in zip(row_ind, col_ind):
            plt.annotate('', xy=truth[j], xytext=found[i], arrowprops=dict(arrowstyle='->', color='r'))

        plt.title(f'Chunk {chunk}\nmax error = {match_dist.max():5f}')
        plt.legend()
        plt.tight_layout() 
    
class Timer:
    def __init__(self, desc:str):
        self.desc = desc
        self.total = 0
        self.start = perf_counter_ns()    
    def reset(self):
        self.start = perf_counter_ns()
    def stop(self):
        self.total += perf_counter_ns() - self.start
    def __str__(self):
        return f'{self.desc} time: {self.total/1e9:.4f} seconds'


def true_distance_matrix(vecs):
    start = perf_counter_ns()
    true_dist = euclidean_distances(vecs, squared=True)
    allpairs_time = perf_counter_ns() - start    
    return true_dist, allpairs_time

def estimated_distance_matrix(pq : ProductQuantization):
    codes = pq.get_compressed_vectors()
    dist_func, is_specialized = pq.get_distance_function(specialized=True)
    assert is_specialized
    npts = len(codes)
    est_dist = np.zeros((npts, npts))
    est_time = 0
    for i in tqdm(range(npts), 'comparing with precalc'):
        start = perf_counter_ns()
        row_dist = dist_func.one2many(codes[i], codes)
        est_time += perf_counter_ns() - start            
        est_dist[i, :] = row_dist
    return est_dist, est_time


def test_distance_function(X, dist_func, prefix):
    t1 = Timer('allpairs')
    allpairs_dist = dist_func.allpairs(X)
    t1.stop()

    npts = len(X)
    m = npts//2
    t2 = Timer('pairwise')
    A = dist_func.pairwise(X[:m], X[:m])
    B = dist_func.pairwise(X[:m], X[m:])
    C = dist_func.pairwise(X[m:], X[:m])
    D = dist_func.pairwise(X[m:], X[m:])
    t2.stop()
    pairwise_dist = np.block([[A, B],
                                [C, D]])
    
    one2many_dist = np.zeros((npts, npts))
    onebyone_dist = np.zeros((npts, npts))
    t3 = Timer('one2many')
    t4 = Timer('distance')
    for i in tqdm(range(npts), prefix+' Computing distance matrix'):            
        t3.reset()
        row = dist_func.one2many(X[i], X)
        t3.stop()
        one2many_dist[i, :] = row

        t4.reset()
        row = [ dist_func.distance(X[i], X[j]) for j in range(npts) ]        
        t4.stop()
        onebyone_dist[i, :] = row

    t5 = Timer('nonsquare')
    dist = dist_func.allpairs_nonsquare(X)
    t5.stop()
    nonsquare_dist = squareform(dist)

    assert np.allclose(allpairs_dist, onebyone_dist)
    assert np.allclose(one2many_dist, onebyone_dist)
    assert np.allclose(pairwise_dist, onebyone_dist)            
    assert np.allclose(nonsquare_dist, onebyone_dist)            
    print(prefix, 'All distance methods agree')

    print(prefix, t1)
    print(prefix, t2)
    print(prefix, t3)
    print(prefix, t4)
    print(prefix, t5)

    return allpairs_dist


def test_compressed_distance_function(compressed_data, pq: ProductQuantization, prefix):
    spec_dist_func, is_specialized = pq.get_distance_function(specialized=True)
    assert is_specialized
    nonspec_dist_func, is_specialized = pq.get_distance_function(specialized=False)
    assert not is_specialized
    X = compressed_data
  
    print(prefix,'optimized distance function')
    specialized = test_distance_function(compressed_data, spec_dist_func, prefix+'  ')
    print(prefix, 'unoptimized distance function')
    unspecialized = test_distance_function(compressed_data, nonspec_dist_func, prefix+'  ')
    assert np.allclose(specialized, unspecialized)

    return specialized

def test_compression(pq : ProductQuantization, vecs, codes, prefix, stddev=None):
    n, d = vecs.shape
    assert len(codes) == n

    print(prefix, f'Testing compression quality: n={n}, d={d}, pq chunks={pq.n_chunks}, chunk clusters={pq.chunk_clusters}')
    
    est_vecs = pq.decompress(codes)
    compressed_vecs = pq.compress(vecs)
    # Test that decompressing many is the same as one at a time
    assert np.allclose(est_vecs, np.array([pq.decompress(codes[i]) for i in range(len(codes))]))
    # Test that compressing many is the same as one at a time
    assert np.allclose(compressed_vecs, np.array([pq.compress(v) for v in vecs]))
    # Test that compress(decompress(code_word)) == code_word 
    assert np.allclose(pq.compress(est_vecs), codes)   
    # Test that decompress(compress(vec)) is near vec
    dists = np.linalg.norm(pq.decompress(compressed_vecs) - est_vecs)
    assert dists.max() < 1e-5
    

    tdist = Timer('euclidean_distances')       
    true_dist = euclidean_distances(vecs, squared=True)
    tdist.stop()
    print(prefix + '  ', tdist)
    
    est_dist = test_compressed_distance_function(compressed_vecs, pq, prefix + '  ')

    compare_matrices('estimatied vs true distances', true_dist, est_dist, prefix)


    compression_error = ((est_vecs-vecs)**2).sum(axis=1)    
    print(prefix, f'mean squared compression error = {compression_error.mean():5f}')

    # How is compression error distributed?
    # make_blobs() clusters are isotropic multi-variate Gaussians at dimension d with same stddev=sigma but different centers.
    # So  x_i - true_center(x)_i is N(0, sigma^2)   and  || x - true_center(x) ||^2 = sigma^2 * ChiSquare(df=d).
    # Assuming PQ has correctly identified subclusters (which is a huge assumption), we have that 
    # sum of || est_x - true_x ||^2 over all chunks should also be distributed as sigma^2 * ChiSquare(df=d)
    if stddev is not None:
        expected_error = d * (stddev**2)    
        print(prefix, f'expected squared compression error = {expected_error:5f}')
        assert abs(compression_error.mean() - expected_error) < stddev

        plt.ecdf(compression_error, label=r'$\Vert x - x_{est} \Vert^2$')
        plt.ecdf(np.random.chisquare(df=d, size=100000)*(stddev**2), label=r'$\chi^2_{%d}$' % d)
        plt.legend()
        plt.tight_layout()



def compare_matrices(desc, actual, estimated, prefix):
    actual = actual.copy()
    estimated = estimated.copy()
    np.fill_diagonal(actual, 1.0)
    np.fill_diagonal(estimated, 1.0)    
    relative_error = np.abs(estimated - actual)/actual    
    assert np.isfinite(relative_error).all()
    print(prefix, f'{desc}')
    print(prefix, f'\tmean relative percentage error (MAPE)={100*relative_error.mean():.2f}%')
    print(prefix, f'\tmax relative percentage error={100*relative_error.max():.2f}%')

def test_clustering():
    n = 1000
    d = 10    
    chunk_bits=4
    nblobs = 2**chunk_bits
    

    def make_clusters(cluster_std):
        X, y, centers = make_blobs(n, d, centers=nblobs, cluster_std=cluster_std, return_centers=True)
        pq = ProductQuantization(d, chunk_dim=2, chunk_bits=chunk_bits, progress_wrapper=tqdm)
        codes = pq.load_and_compress(X, len(X))
        return X, codes, centers, pq

    
    cluster_std = 1e-2    
    X, codes, centers, pq = make_clusters(cluster_std)    
    print(f'cluster centers found correctly: ', end='')
    for found, data, truth in iterate_subvectors(pq, X, centers):
        row_ind, col_ind, match_dist = get_cluster_match(found, truth)        
        assert match_dist.max() < cluster_std
    print('yes')

    test_compression(pq, X, codes, '', cluster_std)

    print(f'showing with more loose clusters')    
    X, codes, centers, pq = make_clusters(0.5)
    show_clustering(X, centers, pq)

    print()



if __name__ == '__main__':
    print(f'Running quick-and-dirty unit tests in {__file__}')

    np.random.seed(5351)

    test_clustering()

    n = 4000
    d = 128
    nblobs = 5
    vecs, labels = make_blobs(n, d, centers=nblobs)
    pq = ProductQuantization(d,  progress_wrapper=tqdm)
    codes = pq.load_and_compress(vecs, len(vecs))
    test_compression(pq, vecs, codes, '')


    n = 10000
    d = 128
    nblobs = 20
    vecs, labels = make_blobs(n, d, centers=nblobs)
    pq = ProductQuantization(d,  progress_wrapper=tqdm)
    codes = pq.load_and_compress(vecs, len(vecs))
    test_compression(pq, vecs, codes, '')

    plt.show()   
