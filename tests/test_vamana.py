from annindex import Vamana
from time import perf_counter_ns
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from tqdm import tqdm


def test_querying(index, vec_query, ground_truth, k, Ls):
    n_query = len(vec_query)

    print()
    print(f'{type(index).__name__}: querying {len(vec_query)} vectors with k={k} and L in {Ls}')
    correct = np.zeros(len(Ls), dtype=int)
    timings = np.zeros((n_query,len(Ls)), dtype=int)
    stats = [ Vamana.QueryStats() for L in Ls ]
    progress = tqdm(range(n_query), desc='querying')
    for i in progress:
        for j, L in enumerate(Ls):
            start = perf_counter_ns()
            result = index.query(vec_query[i], k, L=L, out_stats=stats[j])
            stop = perf_counter_ns()
            predicted = np.array(result)
            correct[j] += len(set(ground_truth[i]).intersection(predicted))
            timings[i, j] = stop-start
        progress.set_postfix_str(f'recall@{k}: {correct[-1]/(k*(i+1)):.3f}')
    recalls = correct/(k*n_query)

    print()
    print('results:')
    print(f'L\trecall@{k:02}\tmean (us)\tP99.9 (us)\tmean hops')
    print('=================================================================')
    for i in range(len(Ls)):        
        print(  f'{Ls[i]:3}      {recalls[i]*100:8.2f}       {timings[:,i].mean()/1000.0:8.1f}         {np.percentile(timings[:,i], q=99.9)/1000.0:8.1f}        {stats[i].nhops/n_query:8.1f}')



if __name__ == '__main__':
    print(f'Running quick-and-dirty unit tests in {__file__}')

    np.random.seed(8352090)

    def true_nearest(data, k):
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(data)
        distances, indices = nn.kneighbors(data)
        return indices
    
    n = 10000
    d = 128
    nblobs = 3
    k = 10
            
    training, _, centers_a = make_blobs(n, d, centers=nblobs, return_centers=True)
    testing, _, centers_b = make_blobs(n, d, centers=nblobs, return_centers=True)
    print('mean distance between centers:', cdist(centers_a, centers_b).mean())

    # Build from first half
    print('building index')
    vnm = Vamana(d, progress_wrapper=tqdm)
    vnm.load(training, len(training))
    vnm.build()

    print('computing ground truth')
    nn = NearestNeighbors(n_neighbors=k).fit(training)
    ground_truth_train = nn.kneighbors(training)[1]
    ground_truth_test = nn.kneighbors(testing)[1]

    # Test on first half, should get exact match
    correct = 0
    errors = []
    pbar = tqdm(range(len(training)), desc='train set')    
    for p in pbar:
        predicted = np.array(vnm.query(training[p], k))
        diff = predicted != ground_truth_train[p]
        if diff.any():
            errors.append(p)
        correct += len(set(ground_truth_train[p]).intersection(predicted))
        pbar.set_postfix_str('recall: {:.3f}'.format(correct/(k*(p+1))))
    print('Recall on train set', correct/(k*len(training)))
    print('Errors in the following indexes:', errors)

    test_querying(vnm, testing, ground_truth_test, k, [10,20,30,40,50,100])


    