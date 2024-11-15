from annindex.vamana import VamanaIndex
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


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
    nblobs = 5
    k = 10
            
    data, labels = make_blobs(n, d, centers=nblobs)
    
    training = data[:n//2]
    testing = data[n//2:]

    # Build from first half
    print('building index')
    vnm = VamanaIndex(d, progress_wrapper=tqdm)
    vnm.build(training)

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

    # Test on second half, compute recall
    correct = 0
    for p in tqdm(range(len(testing)), 'test set'):
        predicted = np.array(vnm.query(testing[p], k))
        correct += len(set(ground_truth_test[p]).intersection(predicted))
    print('Recall on test set', correct/(k*len(testing)))



    