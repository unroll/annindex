from annindex.vamana import VamanaIndex
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


if __name__ == '__main__':
    print(f'Running quick-and-dirty unit tests in {__file__}')


    def true_nearest(data, k):
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(data)
        distances, indices = nn.kneighbors(data)
        return indices
    
    n = 10000
    d = 50
    nblobs = 5
    k = 10
            
    data, labels = make_blobs(n, d, centers=nblobs)
    
    training = data[:n//2]
    testing = data[n//2:]
    nn = NearestNeighbors(n_neighbors=k).fit(training)
    ground_truth_train = nn.kneighbors(training)[1]
    ground_truth_test = nn.kneighbors(testing)[1]
   
    # Build from first half
    print('building index')
    vnm = VamanaIndex(d, R=32, L=70, progress_wrapper=tqdm)
    vnm.build(training)

    
    # Test on first half, should get exact match
    correct = 0
    for p in range(len(training)):
        predicted = np.array(vnm.query(training[p], k))
        diff = predicted == ground_truth_train[p]
        if diff.any():
            print(f'{p}:\tpredicted {predicted}\n\ttrue      {ground_truth_train[p]}')
        correct += len(set(ground_truth_train[p]).intersection(predicted))
    print('Recall on train set', correct/(k*len(training)))

    # Test on second half, compute recall
    correct = 0
    for p in range(len(testing)):
        predicted = np.array(vnm.query(testing[p], k))
        correct += len(set(ground_truth_test[p]).intersection(predicted))
    print('Recall on test set', correct/(k*len(testing)))



    