
import numpy as np
from annindex.pq import ProductQuantization
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from time import perf_counter_ns



if __name__ == '__main__':
    print(f'Running quick-and-dirty unit tests in {__file__}')

    np.random.seed(5351)

    n = 4000
    d = 128
    nblobs = 5

    vecs, labels = make_blobs(n, d, centers=nblobs)

    pq = ProductQuantization(vecs, progress_wrapper=tqdm)
    
    start = perf_counter_ns()
    true_dist = euclidean_distances(vecs, squared=True)
    np.fill_diagonal(true_dist, 1.0)
    allpairs_time = perf_counter_ns() - start

    def compare_matrices(desc, actual, estimated):
        relative_error = np.abs(estimated - actual)/actual    
        assert np.isfinite(relative_error).all()
        print(f'{desc}')
        print(f'\tmean relative percentage error (MAPE)={100*relative_error.mean():.2f}%')
        print(f'\tmax relative percentage error={100*relative_error.max():.2f}%')

    est_dist = np.zeros_like(true_dist)
    est_time = 0
    for i in tqdm(range(len(pq.codes)), 'comparing with precalc'):
        start = perf_counter_ns()
        row_dist = [ pq.distance(pq.get_code(i), pq.get_code(j)) for j in range(len(pq.codes)) ]
        est_time += perf_counter_ns() - start            
        est_dist[i, :] = row_dist 
    np.fill_diagonal(est_dist, 1.0)
    compare_matrices('estimatied (precalc) vs true', true_dist, est_dist)
    
    est2_dist = np.zeros_like(true_dist)
    est2_time = 0
    for i in tqdm(range(len(pq.codes)), 'comparing without precalc'):
        start = perf_counter_ns()
        row_dist = [ pq.distance(pq.get_code(i), pq.get_code(j), allow_precalc=False) for j in range(len(pq.codes)) ]
        est2_time += perf_counter_ns() - start            
        est2_dist[i, :] = row_dist 
    np.fill_diagonal(est2_dist, 1.0)
    compare_matrices('estimatied (precalc) vs estimatied (np precalc)', est2_dist, est_dist)    
    assert np.allclose(est2_dist, est_dist)
    
    print(f'time to compute all {n**2} pair distances (seconds):')
    print(f'\tall pairs using linear algebra:\t{allpairs_time/1e9}')
    print(f'\testimation time (with precalc):\t{est_time/1e9}')
    print(f'\testimation time (no precalc):\t{est2_time/1e9}')
