# annindex

Approximate nearest neighbour search indices implemented in Python, targeted at researchers, small projects, and learners.

## Description

`annindex` provides easy-to-use, easy to understand, reasonably-performing implementations of modern approximate nearest neighbour search (ANNS) indexes.
Modern vector databases (VDBMS) rely on such indexes to provide fast and accurate nearest neighbour queries.

Indexes in annindex are implemented in Python directly from pseudo-code in the relevant papers, aiming at simplicity rather than performance or comprehensive features.
Read the [philosophy](#philosophy-or-why-do-this) behind `annindex` to see why and how this can help researchers, learners, and practitioners. 
If you are going for top performance or production-grade code, try [something else](https://github.com/facebookresearch/faiss).

## Getting Started

The aim is to publish `annindex` on PyPI. once that is achieved, it should be possible to `pip install annindex`.

Unfortunately we are not quite there yet (see [roadmap](#roadmap)), so currently you should check out or otherwise download the code, install the prerequisites yourself, and set up `PYTHONPATH` if needed.

## Usage

```py
import numpy as np                                  # numpy is useful
from tqdm.auto import tqdm                          # display a progress bar
from annindex.vamana import VamanaIndex, QueryStats # import index
from annindex.data import iterate_fvecs             # loader for SIFT data

# Iterate over vectors of SIFT data.
# The iterator does *not* load the entire file -- works one vector at a time.
n_vectors, d, dtype, data_itr = iterate_fvecs('path/to/sift/sift_learn.fvecs')

# Initialize the ANNS index with whatever parameters we want.
# Most indexes accept a tqdm-style function to display progress.
# Many indexes accept an optional tqdm-like function to create a progress bar.
indx = VamanaIndex(d=d, R=32, L=75, progress_wrapper=tqdm)

# Load the data to the index, but do not build the index yet.
# This only reads and loads the first 10000 vectors, even if the file is large.
# In this case, we tell the index to use internally the same precision (dtype)
# as the .fvecs file (FP32), but we could specify different precision.
n = 10000
indx.load(data_itr, n, dtype=dtype)

# Build the index, which can take time.
# If progress_wrapper was passed, it will be used here.
indx.build()

# We can now execute nearest neighbour queries, or get vector by their id.
# This returns the top 5 nearest neighbours to [1,1,1,1,...,1]
q = np.ones(d, dtype=dtype)
k = 5
for i in indx.query(q, k):
   print(f'neighbour {i} = {indx.get(i)}')

# Some indexes have additional per-query parameters that control  performance.
for i in indx.query(q, k, L=200):
   print(f'neighbour {i} = {indx.get(i)}')

# Some indexes support collecting statistics on queries.
stats = QueryStats
result = indx.query(q, k, out_stats=stats)
print(f'found {k} neighbours using {stats.nhops} graph hops')

```
## Documentation

In the [roadmap](#roadmap).
For now, see class, method, and function docstrings.

## Philosophy (or, Why Do This?)

Why do we need `annindex`? 
After all, there are several high quality implementations of ANNS indexes that provide excellent performance, low memory, maximum flexibility, and comprehensive features: [FAISS](https://github.com/facebookresearch/faiss), [hnswlib](https://github.com/nmslib/hnswlib), and [DiskANN](https://github.com/microsoft/DiskANN)), to mention some.

The downside of such quality is that their codebase can be complex: thousands of lines of multi-threaded C++ code interfacing with dependencies to utilize GPUs, multi-core CPUs, disks, and so on.
For example, while the elegant Vamana algorithm at the core of DiskANN can be described in 20 lines of pseudo-code, the DiskANN implementation (which includes much more) is 20K lines of C++.
Moreover, the code can deviate from the paper in surprising ways.
Returning to the DiskANN example, the implementation allows graph edge lists to be larger than R during construction -- unlike the algorithm in the paper.[^1]
Surprisingly, there are few direct, straightforward, Python implementations of these indexes (though Python bindings are usually available).

This presents several problems for anyone who wants to quickly test new ideas, proofs of concept, or simply just learn about vector databases and ANNS:
* Reading the code is hard, due to its complexity.
* Tweaking existing algorithms can be painful. 
  Even very small changes can require days of work to modify dozens of lines of code across several files. 
  Larger changes may require weeks of work.
* It is difficult to compare a brand new algorithm to an existing index, since above implementations are of such high quality. 
  Difference in algorithm performance could be hidden by differences in implementation performance. 
  Research are thus forced to spend significant time creating a fast implementation in C++, or reimplementing the original algorithm.
* Similarly, it is more difficult to compare performance across libraries due to differing quality of implementations and different APIs.
* Because implementations can differ from the published paper in undocumented ways, it is harder to isolate improvements or track down sources of error. 
  Researchers may need to carefully follow the implementation to understand unexpected behaviours.
* Creating high quality implementations of an ANNS index takes significant time and effort. 
  For example, FAISS does not (yet?) have an implementation of [Vamana](https://papers.nips.cc/paper_files/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) (DiskANN), and may never have it.
  This slows research due to the need to use multiple complicated libraries.

[^1]: This is not meant to pick on DiskANN. 
      It is a genuine achievement with a large impact on industry and academia.
      Even studying how the implementation differs from the theory is instructive.
      Nevertheless, the codebase is not conducive to fast experimental modifications nor easy understanding of the algorithm.

### Goals (or, Who Is annindex For?)
`annindex` is meant to serve several audiences: researchers, tinkerers, small project developers, and those trying to learn about vector databases.

First and foremost, `annindex` is meant to address the above issues by creating a small, simple library of important state-of-the-art and classic ANNS indexes that is easy to use with only very common requirements.
Researchers could then use it both as a basis for comparison, and as a way to quickly try out many new ideas.
The code should track the algorithm/pseudocode closely, so that reading it could help them learn how an index works.

`annindex` should be usable for small projects, homework assignments, and implementing simple, non-production VDBMSs.
Perhaps you need a quick, easy to use nearest neighbour search index with few dependencies dependencies on up to 100K vectors? 

Finally, one hopes `annindex` becomes a destination for designers of new ANNS indices.
It should (hopefully) be easy for authors of new indexes to contribute a straightforward Python implementation of their new index to the `annindex` library.
This would make it easier for others to compare to the new work.


### Values (or, How We Reach the Goals)

* Simplicity trumps performance.
  - Code should match the algorithm pseudocode well.
  - Support code (i.e., most of it) should be clear and simple.
  - Avoid very complex data structures.
  - Ideally, implementation should be simple enough to rip out and re-use.

* ... but provide reasonable performance when possible.  
  - Use numpy, sklearn, scipy cleverly for performance.
  - OK to replace naive pseudocode with more optimized loops, as long as this is well-documented in the code. 
    For example, argmin_i f(x_i) inside a pseudocode loop could sometimes be replaced by precalculating f(x_i) for all i's and sorting, thus avoiding an O(n^2) loop.
  - Avoid reading entire file into memory: loads from iterators one vector at a time.
  - Avoid silly wastage: use sum and not Python loops, avoid calling a small function repeatedly
 
* Designed to run in memory, for simplicity *and* performance.
  - Note this does not preclude having disk-based indexes, whether actually stored on disk or in memory.
    Only that there is no attempt to force indexes to operate out-of-core.
  - This does not mean indexes should be *wasteful*. 
    For example, compressors should not save the original data.
      
* Simple, straightforward implementation that people can learn from.
  - Code should be simple and easy to read, even at cost of performance or generality.
    It should be easy to connect code and paper.
  - Well-documented, with numpy-style docstrings and detailed code comments.
  - Code for an index should ideally be contained in a single file.  
  - Shallow and visible class hierarchy that does not take up huge headspace.
    Avoid surprises. 

* Simple to use and extend, even at cost of encapsulation and reusability.
  - Straightforward public APIs.
  - Do not limit extenders.
    For example, implementing a new graph-based index does not *require* deriving from some "GraphIndexBase", a "LoaderMixin", and a "QueryableMixin".[^2]
  - Provides some base classes to help implementors, but they are not required.
    Moreover, base classes currently only offer a very limited private interface for derived classes (*this is not ideal, and may change*).
    Instead, derived classes can manipulate state directly.
  - Do not take this to extreme. Code should still be modular and reusable if it does not harm other goals.

* Minimize deviations from publications. Any such deviations must be contained and documented.
  - For large deviations, include the original approach or algorithm as an option. 
    For example, a multithreaded index construction approach that is materially different than the original approach should be implemented as a separate index class or in a separate function.
  - If deviation is small, include the original algorithm as a comment and explain the changes well.
    For example in the above example of optimizing argmin_i, the output is the same.

* Very limited dependencies.
  - Make it easy to install and use.
  - Main dependencies are numpy, scipy, and scikit-learn. 

[^2]: Joking aside, scikit-learn has been an inspiration for `annindex`. 
    
### Non-goals

`annindex` is not meant for building large production systems, nor is it meant to be "scikit-anns" (perhaps in the future...):
* Not production quality.
* Focus is on simple code, rather than making it very fast or generic.
* Moderate input checks to avoid errors, but assume users generally know what they are doing.
* Not meant to build large systems.
* Not meant to support a huge eco-system.
* Not feature-complete, and is not trying to be.

## Roadmap

- [ ] Add important indexes (without going overboard)
    - [ ] IVF
    - [ ] HNSW
    - [ ] [Indexes on compressed data](https://github.com/unroll/annindex/issues/3)
    - [ ] Composite indexes
    - [ ] Residual compression
    - [ ] IVFOADC+G+P ?   
- [ ] Core index features
    - [ ] Range search
    - [ ] Incremental indexing APIs (insert, delete, update)
    - [ ] Store and load index
- [ ] Advanced indexing
    - [ ] Store metadata with vectors
    - [ ] Filtered (predicated) queries
    - [ ] Hybrid (multi-vector) queries
- [ ] Quality of life
    - [ ] Testing    
    - [ ] Package documentation
    - [ ] Better APIs (access parameters, internal APIs)
- [ ] PyPI release

## Contributing

Bug reports and suggestions are useful -- go ahead and open an [issue](https://github.com/unroll/annindex/issues).
No template yet so try to be nice about it.

Code contributions that help progress the roadmap or implementations of new SotA indexes are greatly appreciated!
Please read our [philosophy](#philosophy-or-why-do-this) about what the code should look like, and why.
Go ahead and fork the repo, create and commit to a feature branch, and create a pull request once ready.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feat/xx/your-feature-name`)
3. Commit your Changes (`git commit -m 'Adding XYZ'`)
4. Push to the Branch (`git push origin feat/xx/your-feature-name`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Moshe Gabel: mgabel (-at-) cs.toronto.edu

Project Link: [https://github.com/unroll/annindex](https://github.com/unroll/annindex)

