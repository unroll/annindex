import numpy as np
from numpy.typing import NDArray
from typing import IO, Iterator
import os

def _get_fvecs_dtypes(filename: str, scalar_type: str = 'auto') -> tuple[np.dtype, np.dtype]:
    """Return python dtypes suitable for reading fvecs file.

    Parameters
    ----------
    filename : str
        Name of file to read.
    scalar_type : str, optional
        Type of vector scalars: `fvecs` (float32), `ivecs` (int32), `bvecs` (uint8), or `auto` (default) to determine using the file extension.

    Returns
    -------
    np.dtype
        Numpy dtype for length of vector.
    np.dtype
        Numpy dtype for vector elements (scalars).    
    """    
    if scalar_type.lower() == 'auto':
        extension = os.path.splitext(filename)[1]
        scalar_type = extension[1:] # skip dot (.)
    # Select type of data in file
    mapping = {'fvecs':np.dtype('<f4'),
               'ivecs':np.dtype('<i4'),
               'bvecs':np.dtype('<u1')}
    if scalar_type not in mapping:
         raise ValueError(f'Cannot load file {filename} with unknown scalar type {scalar_type}. Try one of {list(mapping.keys())}.')
    dtype = mapping[scalar_type]
    dimension_dtype = np.dtype('<i4') # little-endian, int32     
    return dimension_dtype, dtype

def load_fvecs(filename: str, scalar_type: str = 'auto') -> NDArray:
    """
    Read and return the content of .fvecs, .ivecs, or .bvecs file as a 2D array.

    See http://corpus-texmex.irisa.fr/ for file format.    
    
    Note this function opens the file for reading twice. 
    There is technically a window of time where the file could be renamed, changed, or deleated.    

    Parameters
    ----------
    filename : str
        Name of file to read.
    scalar_type : str, optional
        Type of vector scalars: `fvecs` (float32), `ivecs` (int32), `bvecs` (uint8), or `auto` (default) to determine using the file extension.

    Returns
    -------
    out : ndarray
        Contents of file as a 2D array.
        
    """    
    dimension_dtype, dtype = _get_fvecs_dtypes(filename, scalar_type)
    # Read vector dimension first
    d = np.fromfile(filename, dtype=dimension_dtype, count=1)[0]
    # Use numpy fields to read and return array
    x = np.fromfile(filename, dtype=[('d', dimension_dtype), ('vecs', dtype, (d,))])
    # Convert to native byte order, ideally with no casting or copying
    native_dtype = dtype.newbyteorder('=')
    return x['vecs'].astype(native_dtype, casting='same_kind', copy=False)
        

def iterate_fvecs(filename: str, scalar_type: str = 'auto') -> tuple[int, int, np.dtype, Iterator[NDArray],]:
    """
    Read and return the content of .fvecs, .ivecs, or .bvect one vector at a time.

    See http://corpus-texmex.irisa.fr/ for file format.
    
    Note this function opens the file for reading twice. 
    There is technically a window of time where the file could be renamed, changed, or deleated.

    Parameters
    ----------
    filename : str
        Name of file to read.
    scalar_type : str, optional
        Type of vector scalars: `fvecs` (float32), `ivecs` (int32), `bvecs` (uint8), or `auto` (default) to determine using the file extension.

    Returns
    -------
    int
        Number of vectors in file.
    int
        Dimension of each vector
    dtype
        Numpy dtype of vector elements
    Iterator
        An iterator that yields vectors one at a time from the file as numpy arrays.
        
    See Also
    --------
    load_fvecs : Load file directly as numpy 2D array

    Examples
    --------

    The following example uses numpy `fromiter` to generate a 2D array
    
    >>> nvectors, d, dtype, gen = iterate_fvecs(filename)
    ... x = np.fromiter(gen, dtype=(dt, d), count=n)    
    """        
    # Set file type from extension
    if scalar_type.lower() == 'auto':
        extension = os.path.splitext(filename)[1]
        scalar_type = extension[1:] # skip dot (.)
    # Select type of data in file
    mapping = {'fvecs':np.dtype('<f4'),
               'ivecs':np.dtype('<i4'),
               'bvecs':np.dtype('<u1')}
    if scalar_type not in mapping:
         raise ValueError(f'Cannot load file {filename} with unknown scalar type {scalar_type}. Try one of {list(mapping.keys())}.')
    dtype = mapping[scalar_type]
    native_dtype = dtype.newbyteorder('=')
    dimension_dtype = np.dtype('<i4') # little-endian, int32        
    with open(filename, 'rb') as f:        
        # Read dimension of first vector
        d = np.fromfile(f, dtype=dimension_dtype, count=1)[0]
        if d <= 0:
            raise RuntimeError('Vectors with dimension {d} in file {filename}')
        # Estimate number of vectors from file size
        nbytes = os.fstat(f.fileno()).st_size
        vector_bytes = 4 + d * dtype.itemsize 
        if nbytes % vector_bytes != 0:            
            raise RuntimeError(f'File {filename} with size {nbytes}, dimension {d}, and type {dtype} does not have an integer number of vectors. Aborting.')        
        num_vectors = nbytes // vector_bytes        
    
    # Create iterator that yields vectors          
    def iterator_impl():        
        """Yields vectors in file, after casting to native endianess."""
        # We are opening the file twice, 
        with open(filename, 'rb') as f:
            # Keep reading 4 bytes (int32 of dimension) and vector until done
            while len(f.read(4)) == 4:
                # Read next vector
                x = np.fromfile(f, dtype=dtype, count=d)
                # Convert to native byte order, ideally with no casting or copying
                yield x.astype(native_dtype, casting='same_kind', copy=False)
    return num_vectors, d, native_dtype, iterator_impl() 
