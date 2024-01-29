import numpy as np
from scipy.linalg import lu,qr

def lu_full(a):
    """
    LU decomposition with full pivoting.

    Returns:
    ind: pivot indices.
    """
    b = a.astype(float)
    n, r = b.shape
    ind = np.zeros(r, dtype=int)

    for i in range(r):
        big_ind = np.argmax(np.abs(b))
        i0, j0 = np.unravel_index(big_ind, (n, r))
        b -= np.outer(b[:, j0], b[i0, :]) / b[i0, j0]
        ind[i] = i0

    return ind


def maxvol2(a, ind=None, do_qr=False, eps=5e-2, niters=100, do_lu_full=False):
    '''
    Maximal volume submatrix in an tall matrix
    maxvol2(a, ind=None) : Computes maximal volume submatrix in A starting from LU
    maxvol2(a, ind) : Computes maximal volume submatrix in A starting from ind
    
    Returns:
    rows indices that contain maximal volume submatrix
    '''
    n, r = a.shape
    if n <= r:
        return list(range(n))

    # Handling optional arguments
    if do_qr:
        a, _ = qr(a, mode='economic')

    if do_lu_full:
        ind = lu_full(a)
    else:
        # Perform LU decomposition
        # Results different from Matlab
        ######### ??????
        p_matrix, l_dmp, u_dmp = lu(a)

        # Convert the permutation matrix to a permutation vector
        p = np.argmax(p_matrix.T, axis=1) 

        ind = p[0:r]
        
    sbm = a[ind, :]
    b = np.linalg.solve(sbm, a.T).T  # Solve for X in sbm * X = a

    iter = 0
    while iter <= niters:
        mx0, big_ind = np.max(np.abs(b)), np.argmax(np.abs(b))
        i0, j0 = np.unravel_index(big_ind, (n, r))

        if mx0 <= 1 + eps:
            return sorted(ind)

        k = ind[j0]
        b += np.outer(b[:, j0], (b[k, :] - b[i0, :])) / b[i0, j0]
        ind[j0] = i0
        iter += 1

    return sorted(ind)

def tt_ind2sub(siz, idx):
    """
    Convert a linear index to multi-dimensional subscript indices for a tensor.

    Args:
    siz: The size of each dimension of the tensor.
    idx: The linear index (or an array of linear indices).

    Returns:
    ind: A 2D array where each row contains the subscript indices 
         corresponding to the linear index in `idx`.
    """
    n = len(siz)
    m = np.array(siz).reshape(1, n)
    m = m[:, :n-1]
    k = np.concatenate(([1], m.flatten()))
    k = np.cumprod(k)
    ind = idx - 1
    ind = np.tile(ind, (n, 1)).T
    k = np.tile(k, (len(ind), 1))
    ind = np.floor_divide(ind, k)
    m = np.tile(m, (len(ind), 1))
    ind[:, :n-1] = ind[:, :n-1] - ind[:, 1:n] * m

    # Don't need the following because we adjust for zero-based indexing.
    # ind = ind + 1

    return ind

def localcross(Y, tol):
    """
    Full-pivoted cross for truncating one ket TT block instead of SVD.

    Args:
    Y: Input tensor.
    tol: Tolerance level.

    Returns:
    u: Resulting matrix U.
    v: Resulting matrix V.
    I: Indices of pivot elements.
    """
    # n, m, b = Y.shape
    b, n, m = Y.shape
    
    minsz = min(n, m * b)
    u = np.zeros((n, minsz))
    v = np.zeros((minsz, m * b))
    res = Y.reshape(n, m * b)

    # Return also the indices
    I = np.zeros(minsz, dtype=int)
    val_max = np.max(np.abs(Y))

    for r in range(minsz):
        res = res.ravel(order='F')
        val, piv = np.max(np.abs(res)), np.argmax(np.abs(res))
        piv = tt_ind2sub([n, m * b], piv + 1)  # Adjust for zero-based index
        if val <= tol * val_max:
            break

        res = res.reshape((n, m * b), order='F')
        u[:, r] = res[:,piv[0][1]]
        v[r, :] = res[piv[0][0], :] / res[piv[0][0], piv[0][1]]
        res = res - np.outer(u[:, r], v[r, :])
        I[r] = piv[0][0]

    # Return indices
    # In MATLAB, the expression r = find(I == 0, 1); is used to find the index of the first occurrence of 0 in the array I. 
    # But what's the point of doing it ???????
    r = np.where(I == -1)[0] ####### just as a placeholder
    if r.size == 0:
        r = minsz
    else:
        r = r[0]  # Get the first occurrence
    I = I[:r]
    u = u[:, :r]
    v = v[:r, :]

    if r == 0:
        u = np.zeros((n, 1))
        v = np.zeros((1, m * b))
        I = np.array([1])

    # QR decomposition of u, in case we don't have enrichment
    u, rv = np.linalg.qr(u, mode='reduced')
    v = rv @ v

    return u, v, I