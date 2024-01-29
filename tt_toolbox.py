import numpy as np
from tt_tensor import *
from scipy.linalg import qr

def tt_mem(tt):
    """
    Memory required to store the tensor TT.
    
    Parameters:
    tt: list of numpy arrays
        The TT-tensor whose memory usage is to be calculated.
    
    Returns:
    int: The total number of elements in the TT-tensor.
    """
    mem = 0
    d = len(tt)
    for k in range(d):
        mem += tt[k].size
    return mem

def tt_size(tt):
    """
    Mode dimensions of a TT-tensor in TT1.0 format.
    This function returns the mode dimensions of a TT tensor. 
    
    Parameters:
    tt: list of numpy arrays
        The TT-tensor whose mode dimensions are to be calculated.
    
    Returns:
    sz: A list containing the mode dimensions of the TT-tensor.
    """
    # Get the number of tensors in the tensor train
    d = len(tt)

    # Initialize an empty list to store the mode dimensions
    sz = [0] * d

    # Iterate over each tensor and get its mode dimension
    for i in range(d):
        sz[i] = tt[i].shape[0]

    return sz

def tt_meshgrid_vert(list_of_tensors):
    """
    Analogue of the meshgrid function for the TT-format,
    concatenating tensor trains with ones using tkron.
    Computes the meshgrid based on "1d" representations.

    Parameters:
    args: A list of 1D tt_tensors representing different dimensions.

    Returns:
    X: A list of tt_tensors representing the meshgrid.
    """
    
    X = list_of_tensors
    d = len(X)  # Number of dimensions

    # Concat all n to the common storage
    newn = []
    pos = [1]
    for i in range(d):
        newn.append(X[i].n)
        pos.append(pos[i] + X[i].d)

    # Expand
    for i in range(d):
        if i > 0:
            expand = tt_ones(newn[:pos[i]-1][0])
            X[i] = tkron(expand, X[i])

        if i < d - 1:
            expand = tt_ones(newn[pos[i+1]-1:][0])
            X[i] = tkron(X[i], expand)

    return X

def tt_ones(n):
    """
    Creates a tt-tensor of ones.

    Input:
    n: The size of the tensor.

    Returns:
    tt_tensor: A tt-tensor of ones with the specified size.
    """
    # Create an array of ones
    tt = np.ones((n, 1))

    return tt_tensor(tt)


def tkron(a, b):
    """
    Kronecker product of two TT-tensors, little-endian tensor type.

    Parameters:
    a, b (tt_tensor): Two TT-tensors to be multiplied.

    Returns:
    tt_tensor: Resulting TT-tensor after Kronecker product.
    """
    if not a.core.size:
        return b
    elif not b.core.size:
        return a

    c = tt_tensor()
    c.d = a.d + b.d
    c.core = np.concatenate([a.core, b.core])
    c.n = np.array([a.n, b.n])
    c.r = np.concatenate([a.r[:a.d], b.r])

    c.ps = np.cumsum(np.insert(c.n * c.r[:-1] * c.r[1:], 0, 1))

    return c

def dot(tt1, tt2, chunk_start=None, chunk_end=None, do_qr=False):
    # Handle different number of arguments
    if chunk_start is not None and isinstance(chunk_start, bool):
        do_qr = chunk_start
        chunk_start = None
        chunk_end = None

    if chunk_start is not None:
        if tt2.d <= tt1.d:
            raise ValueError('Chunky dot is defined only if tt2.d > tt1.d')

        # Apply dot to a chunk of tt2
        tt2_chunk1 = chunk(tt2, chunk_start, chunk_end)
        D = dot(tt1, tt2_chunk1, do_qr)
        D = np.reshape(D, (tt2.r[chunk_start], tt2.r[chunk_end + 1]))
        p = []

        if chunk_start > 1:
            tt2_chunk1 = chunk(tt2, 1, chunk_start - 1)
            p = tt2_chunk1 @ D

        if chunk_end < tt2.d:
            tt2_chunk2 = chunk(tt2, chunk_end + 1, tt2.d)
            if not p:
                p = D @ tt2_chunk2
            else:
                p = tkron(p, tt2_chunk2)

        return p

    # Perform QR decomposition if requested
    if do_qr:
        tt1, rv1 = qr(tt1, mode='lr')
        tt2, rv2 = qr(tt2, mode='lr')

    d = tt1.d
    r1 = tt1.r
    r2 = tt2.r
    ps1 = tt1.ps
    ps2 = tt2.ps
    n = tt1.n
    core1 = tt1.core
    core2 = tt2.core

    p = np.eye(r1[0] * r2[0])
    p = np.reshape(p, (r1[0] * r2[0] * r1[0], r2[0]))

    for i in range(d):
        cr1 = core1[ps1[i]:ps1[i + 1]]
        cr2 = core2[ps2[i]:ps2[i + 1]]
        cr2 = np.reshape(cr2, (r2[i], n[i] * r2[i + 1]))

        p = p @ cr2
        p = np.reshape(p, (r1[0] * r2[0], r1[i] * n[i], r2[i + 1]))
        p = np.transpose(p, (0, 2, 1))
        p = np.reshape(p, (r1[0] * r2[0] * r2[i + 1], r1[i] * n[i]))

        cr1 = np.reshape(cr1, (r1[i] * n[i], r1[i + 1]))

        p = p @ np.conj(cr1)
        p = np.reshape(p, (r1[0] * r2[0], r2[i + 1], r1[i + 1]))
        p = np.transpose(p, (0, 2, 1))
        p = np.reshape(p, (r1[0] * r2[0] * r1[i + 1], r2[i + 1]))

    # Finalize according to QR decomposition
    if do_qr:
        r2old = rv2.shape[1]
        r1old = rv1.shape[1]
        p = p @ rv2
        p = np.reshape(p, (r1[0] * r2[0], r1[d], r2old))
        p = np.transpose(p, (0, 2, 1))
        p = np.reshape(p, (r1[0] * r2[0] * r2old, r1[d]))
        p = p @ np.conj(rv1)
        p = np.reshape(p, (r1[0], r2[0], r2old, r1old))
        p = np.transpose(p, (0, 1, 3, 2))
        if r1[0] * r2[0] == 1:
            p = np.reshape(p, (r1old, r2old))
    else:
        p = np.reshape(p, (r1[0], r2[0], r1[d], r2[d]))
        if r1[0] * r2[0] == 1:
            p = np.reshape(p, (r1[d], r2[d]))

    return p

###### Need to debug based on that tensor n is not a scalar
def chunk(b, i, j):
    '''
    Cut the (i,j) part out of the TT-tensor
    '''

    if i > j:
        i, j = j, i

    ps = b.ps[i:j+1]
    r = b.r[i:j+1]
    n = b.n[i:j]
    cr = b.core[ps[0]:ps[-1]]

    # Adjust ps to be relative to the new start
    ps = [p - ps[0] + 1 for p in ps]

    # Update the properties of the tensor
    b.r = r
    b.n = n
    b.ps = ps
    b.core = cr
    b.d = len(n)

    return b