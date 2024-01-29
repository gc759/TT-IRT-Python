import numpy as np

from tt_tensor import *
from tt_toolbox import *
from other_toolbox import *

# Still need to define:
# core2cell
# don't really understand why using this and the reshape stuff
# cell2core
# will have another look when testing 

# check dividing stuff

def sample_others_nested(YX, X, ind, ry_prev, dir):
    """
    Sample TT tensor X at nested indices ind taken from y
    X, YX are 1 x nx cell arrays, ry_prev = ry(i) for dir>0 and ry(i+1) otherwise.
    """
    nx = len(X)
    
    if dir > 0:
        for j in range(nx):
            rx1, n, rx2 = X[j].shape
            YX[j] = np.dot(YX[j], X[j].reshape(rx1, n * rx2))
            YX[j] = YX[j].reshape(-1, ry_prev * n, rx2)  # rx(1,j)
            YX[j] = YX[j][:, ind, :]
            YX[j] = YX[j].reshape(-1, rx2)  # rx(1,j)*ry(i+1)
    else:
        for j in range(nx):
            rx1, n, rx2 = X[j].shape
            YX[j] = np.dot(X[j].reshape(rx1 * n, rx2), YX[j])
            YX[j] = YX[j].reshape(rx1, n * ry_prev, -1)  # rx(d+1,j)
            YX[j] = YX[j][:, ind, :]
            YX[j] = YX[j].reshape(rx1, -1)  # ry(i)*rx(d+1,j)

    return YX

def sample_others_indep(YX, X, ind, dir):
    """
    Sample TT tensor X at independent indices ind of length nq
    X, YX are 1 x nx cell arrays
    """
    nx = len(X)
    nq = len(ind)

    if dir > 0:
        for j in range(nx):
            rx1, _, rx2 = X[j].shape # placeholder used to ignore the second output of the shape
            YX[j] = YX[j].reshape(-1, nq, rx1)  # rx(1,j)
            YXnew = np.zeros((YX[j].shape[0], nq, rx2))

            for k in range(nq):
                YXnew[:, k, :] = np.dot(YX[j][:, k, :].reshape(-1, rx1), 
                                        X[j][:, ind[k], :].reshape(rx1, rx2))  # rx(1,j)

            YXnew = YXnew.reshape(-1, rx2)  # rx(1,j)*nq
            YX[j] = YXnew

    else:
        for j in range(nx):
            rx1, _, rx2 = X[j].shape
            YX[j] = YX[j].reshape(rx2, nq, -1)  # rx(d+1,j)
            YXnew = np.zeros((rx1, nq, YX[j].shape[2]))

            for k in range(nq):
                YXnew[:, k, :] = np.dot(X[j][:, ind[k], :].reshape(rx1, rx2), 
                                        YX[j][:, k, :].reshape(rx2, -1))  # rx(d+1,j)

            YXnew = YXnew.reshape(rx1, -1)  # nq*rx(d+1,j)
            YX[j] = YXnew

    return YX

def qrmaxvol_block(yl, yr, dir, z=None):
    """
    Move non-orth center between the blocks, computing enrich, QR and maxvol
    """
    r0, nl, _, bl = yl.shape
    _, nr, r2, br = yr.shape

    if dir > 0:
        # Reshape all
        yl = yl.reshape(r0 * nl, -1)
        r1 = yl.shape[1]
        if z is not None:
            # Enrich
            yl = np.concatenate([yl, z], axis=1)

        # QR
        yl, rv = np.linalg.qr(yl, mode='reduced')
        rv = rv[:, :r1]

        # Maxvol and divide
        ind = maxvol2(yl)  # maxvol2 function needs to be defined separately with a Python version
        YY = yl[ind, :]
        yl = np.linalg.solve(YY.T, yl.T).T

        # Update r
        r1 = yl.shape[1]
        yl = yl.reshape(r0, nl, r1, bl)

        # Cast non-orths
        if yr.size > 0:
            rv = YY @ rv
            yr = yr.reshape(-1, nr * r2 * br)
            yr = rv @ yr
            yr = yr.reshape(r1, nr, r2, br)
    else:
        # Reshape all
        yr = yr.reshape(-1, nr * r2)
        r1 = yr.shape[0]
        if z is not None:
            # Enrich
            yr = np.concatenate([yr, z.T], axis=0)

        # QR
        yr = yr.T
        yr, rv = np.linalg.qr(yr, mode='reduced')
        rv = rv[:, :r1]

        # Maxvol and divide
        # Replace maxvol2 function with its Python equivalent
        ind = maxvol2(yr)  # maxvol2 function needs to be defined separately
        yr = yr.T
        YY = yr[:, ind]
        yr = np.linalg.solve(YY, yr)

        # Update r
        r1 = yr.shape[0]
        yr = yr.reshape(r1, nr, r2, br)

        # Cast non-orths
        if yl.size > 0:
            rv = rv.T @ YY
            yl = yl.reshape(-1, bl)
            yl = yl.T
            yl = yl.reshape(bl * r0 * nl, -1)
            yl = yl @ rv
            yl = yl.reshape(bl, r0 * nl * r1)
            yl = yl.T
            yl = yl.reshape(r0, nl, r1, bl)

    return yl, yr, r1, ind

def indexmerge(*args):
    """
    Merges two or three indices in the little-endian manner
    """
    sz1 = max(args[0].shape[0], 1)
    sz2 = max(args[1].shape[0], 1)
    sz3 = 1

    if len(args) > 2:  # Currently allows only 3
        sz3 = max(args[2].shape[0], 1)

    # J1 goes to the fastest index, just copy it
    J1 = np.tile(args[0], (sz2 * sz3, 1))

    # J2 goes to the middle
    J2 = args[1].reshape(1, -1)
    J2 = np.tile(J2, (sz1, 1))  # now sz1 ones will be the fastest
    J2 = J2.reshape(sz1 * sz2, -1)
    J2 = np.tile(J2, (sz3, 1))

    J = np.concatenate((J1, J2), axis=1)

    if len(args) > 2:
        # J3 goes to the slowest
        J3 = args[2].reshape(1, -1)
        J3 = np.tile(J3, (sz1 * sz2, 1))  # now sz1 ones will be the fastest
        J3 = J3.reshape(sz1 * sz2 * sz3, -1)

        J = np.concatenate((J, J3), axis=1)

    return J

def evaluate_fun(i, Jl, Jr, n, ifun, ffun, crX, rx, YXl, YXr, vec, ievalcnt, fevalcnt):
    """
    Evaluate the user function at cross indices
    """
    cry = None

    if ifun is not None:
        J = indexmerge(Jl[i], np.arange(1, n[i] + 1)[:, None], Jr[i + 1])
        if vec:
            cry = ifun(J)
        else:
            # We need to vectorize the function
            cry = ifun(J[0, :])
            b = len(cry)
            cry = np.reshape(cry, (1, b))
            cry = np.vstack([cry, np.zeros((len(J) - 1, b))])
            for j in range(1, len(J)):
                cry[j, :] = ifun(J[j, :])

        ievalcnt += len(J)

    if ffun is not None:
        ry1 = YXl[i][0].shape[0] // rx[0, 0] # seems like we need a integer here, do double check when testing
        ry2 = YXr[i + 1][0].shape[1] // rx[-1, 0]
        nx = len(crX)
        # Compute the X at Y indices
        fx = np.zeros((ry1 * n[i] * ry2, np.sum(rx[0, :] * rx[-1, :])))
        pos = 0
        for j in range(nx):
            cr = crX[i][j]
            cr = cr.reshape(rx[i, j], n[i] * rx[i + 1, j])
            cr = YXl[i][j] @ cr
            cr = cr.reshape(rx[0, j] * ry1 * n[i], rx[i + 1, j])
            cr = cr @ YXr[i + 1][j]
            cr = cr.T.reshape(ry1 * n[i] * ry2, rx[-1, j] * rx[0, j])
            fx[:, pos:pos + rx[-1, j] * rx[0, j]] = cr
            pos += rx[-1, j] * rx[0, j]

        # Call the function
        fevalcnt += ry1 * n[i] * ry2
        if vec:
            fy = ffun(fx)
        else:
            fy = ffun(fx[0, :])
            b = len(fy)
            fy = np.reshape(fy, (1, b))
            fy = np.vstack([fy, np.zeros((ry1 * n[i] * ry2 - 1, b))])
            for j in range(1, ry1 * n[i] * ry2):
                fy[j, :] = ffun(fx[j, :])

        if ifun is None:
            cry = fy
        else:
            cry += fy

    return cry, ievalcnt, fevalcnt


def truncate_block(yl, yr, tol, dir):
    """
    Truncate a block via the full cross
    """
    y_trunc = None

    if dir > 0:
        # Reshape it
        r0, nl, r1, b = yl.shape
        yl = yl.reshape(r0 * nl, -1)  # remaining dimensions may contain b

        if tol > 0:
            # Full-pivot cross should be more accurate
            yl, rv = localcross(yl, tol) # localcross function needs to be implemented in Python
        else:
            yl, rv = np.linalg.qr(yl, mode='reduced')

        y_trunc = np.dot(yl, rv).reshape(r0, nl, r1, b)
        rv = rv.reshape(-1, b).T
        rv = rv.reshape(-1, r1)
        r1 = yl.shape[1]
        yl = yl.reshape(r0, nl, r1)

        if yr.size > 0:
            _, nr, r2 = yr.shape
            yr = yr.reshape(-1, nr * r2)
            yr = np.dot(rv, yr)
            yr = yr.T.reshape(b, r1 * nr * r2).T
            yr = yr.reshape(r1, nr, r2, b)
    else:
        # Reshape it
        r1, nr, r2, b = yr.shape
        yr = yr.reshape(r1, -1).T
        yr = yr.reshape(nr * r2, b * r1)

        if tol > 0:
            # Full-pivot cross should be more accurate
            yr, rv = localcross(yr, tol) # localcross function needs to be implemented in Python
        else:
            yr, rv = np.linalg.qr(yr, mode='reduced')

        y_trunc = np.dot(yr, rv).T.reshape(nr * r2 * b, r1).T
        y_trunc = y_trunc.reshape(r1, nr, r2, b)
        rv = rv.reshape(-1, r1)
        r1 = yr.shape[1]
        yr = yr.T.reshape(r1, nr, r2)

        if yl.size > 0:
            r0, nl, _ = yl.shape
            yl = yl.reshape(r0 * nl, -1)
            yl = np.dot(yl, rv.T)
            yl = yl.reshape(r0, nl, r1, b)

    return yl, yr, r1, y_trunc

def project_solution_to_residual(cry, dir, ZY1, ZY2):
    """
    Project the solution cry onto residual bases given by ZY1, ZY2.
    Returns: crz: for residual update, crs: for enrichment
    Input and output are r1-n-r2-b arrays
    """
    ry1, n, ry2, b = cry.shape

    if dir > 0:
        crs = cry.reshape(ry1 * n * ry2, b).T
        crs = crs.reshape(b * ry1 * n, ry2)
        crs = crs @ ZY2
        crs = crs.reshape(b, -1).T
        crs = crs.reshape(ry1, n, -1, b)  # rz(i+1)
        
        crz = crs.reshape(ry1, -1)
        crz = ZY1 @ crz
        rz1 = ZY1.shape[0]
        crz = crz.reshape(rz1, n, -1, b)
    else:
        crs = cry.reshape(ry1, n * ry2 * b)
        crs = ZY1 @ crs
        crs = crs.reshape(-1, n, ry2, b)  # rz(i)
        
        crz = crs.reshape(-1, b).T
        crz = crz.reshape(-1, ry2)
        crz = crz @ ZY2
        crz = crz.reshape(b, -1).T
        rz2 = ZY2.shape[1]
        crz = crz.reshape(-1, n, rz2, b)

    return crz, crs

def truncate_residual(crz, dir, kickrank, tol_local, ry, expand):
    """
    Truncates the residual depending on kickrank
    Input: r1-n-r2-b tensor
    Output: always m-r matrix where m = n*other_rank
    """
    rz1, n, rz2, b = crz.shape

    if dir > 0:
        crz = crz.reshape(rz1 * n, rz2 * b)
    else:
        crz = crz.reshape(rz1 * n * rz2, b).T
        crz = crz.reshape(b * rz1, n * rz2).T

    nrmz = np.linalg.norm(crz, 'fro')
    if nrmz == 0:
        crz = np.random.randn(*crz.shape)
    else:
        crz = crz / nrmz  # prevent underflows

    if abs(kickrank - round(kickrank)) < 1e-8:
        new_rank = kickrank  # kickrank is an exact integer value
    else:
        new_rank = int(np.ceil(kickrank * ry))  # kickrank is a fraction of the solution rank

    if crz.shape[1] > new_rank:
        # Truncate crz if it's too large
        crz, _ = localcross(crz, tol_local) # localcross function needs to be implemented in Python
        crz = crz[:, :min(crz.shape[1], new_rank)]
    elif expand:
        # Expand crz up to kickrank*ry if necessary
        crz = np.hstack([crz, np.random.randn(crz.shape[0], new_rank - crz.shape[1])])
        crz, _ = np.linalg.qr(crz, mode='reduced')

    return crz

def amen_cross_s(inp, fun, tol, y=4, nswp=20, kickrank=4, verb=1, 
                 vec=True, exitdir=0, tol_exit=None, stop_sweep=0, 
                 dir=1, auxinp=[], auxfun=[], 
                 sr=False, lr=False, sm=False, lm=False, si=False, li=False):
    """
    Block cross with error-based enrichment ("S": stabilized, statistics).

    Implements block cross with error-based enrichment for tensor train (TT) interpolation. It interpolates functions
    either defined by indices or dependent on other TT tensors. It can compute min, max values, and return maxvol indices.

    Parameters:
    - inp: Mode sizes as a column vector or a cell array of TT tensors.
    - fun: The function to interpolate. It can be an index-defined function (fun(ind)) or an elementwise function depending on other TT tensors (fun(x)).
    - tol: Tolerance for convergence.

    Optional parameters
    - y: Initial approximation. Can be:
        - a tt_tensor: in this case maxvol indices are computed.
        - an integer value: number of uniformly random indices. This is the default value of 4.
        - an integer M x d array of particular index values.
        - a 1 x d cell array of nested cross indices Jy, which must match the direction of the warm-up sweep (see dir, exitdir and Jy below).
    - nswp: Maximal number of sweeps. Default is 20.
    - kickrank: Error/enrichment rank. Default is 4.
    - verb: Verbosity level. Default is 1.
    - vec: If the function can accept and return vectorized values. Default is TRUE.
    - tol_exit: Stopping tolerance. Default is equal to tol.
    - exitdir: Direction control for the last sweep. Default is 0.
    - dir: Direction of the first computing sweep. Default is 1.
    - auxinp: Secondary input data.
    - auxfun: Secondary input function.

    Statistical parameters:
    - Statistical parameters for estimating min and max values (e.g., 'sr', 'lr', 'sm', 'lm', 'si', 'li').

    Returns:
    - y: Interpolated tensor.
    - statvals: Estimated min and max values.
    - statind: Corresponding indices.
    - Jy: Maxvol indices.
    - evalcnt: Total number of function evaluations.
    """
    
    # Set the default exit tolerance if none is provided
    if tol_exit is None:
        tol_exit = tol

    # Process the additional statistical parameters. Initialize the set of soughts, adding statistical parameters if they are set to True
    soughts = set()
    if sr: soughts.add('sr')
    if lr: soughts.add('lr')
    if sm: soughts.add('sm')
    if lm: soughts.add('lm')
    if si: soughts.add('si')
    if li: soughts.add('li')

    # We need all those empty guys to call subroutines uniformly. Cf. NULL in C
    X = []
    rx = []
    ifun = []
    ffun = []
    n = np.ones(1)

    # Distinguish general and TT-fun inputs
    if isinstance(inp, list):
        # Need to modify after we have the tensor functions in python.
        # Check if the first input is a list of TT (Tensor Train) tensors.
        # Expecting input in the form of a list containing tensor(s): [tensor1, tensor2, ...]
        # First input is an array of TT tensors
        # Input being list of tensor: [tensor] ??
        X = inp
        ffun = fun
    elif isinstance(inp, np.ndarray):
        # Check if the first input is an array representing mode sizes.
        # Expecting input as a NumPy array detailing mode sizes: np.array([size1, size2, ...])
        ifun = fun
        n = inp

    if auxinp and auxfun:
        if isinstance(auxinp, list):
            # Check if the second input is an array of TT tensors
            if not ffun:
                X = auxinp
                ffun = auxfun
            else:
                raise ValueError("Cannot use ffun on both inputs")
        else:
            if not ifun:
                ifun = auxfun
                n = auxinp
            else:
                raise ValueError("Cannot use ifun on both inputs")

    YX = [None] * 1
    ZX = [None] * 1


    # If there is a TT-fun part, prepare it for computations
    if X:
        nx = len(X)
        d = X[0].d
        n = X[0].n
        rx = np.zeros((d + 1, nx))
        X = [np.reshape(X, (1, nx))] 
        X = [X[0]] + [None] * (d - 1)
        X.extend([None] * (d-1))  # Add empty cells to X
        for i in range(nx):
            rx[:, i] = X[0][i].r
            X[i] = core2cell(X[0][i])

        YX = [[np.eye(rx[0, j])] for j in range(nx)]
        ZX = [[np.eye(rx[0, j])] for j in range(nx)]

        for j in range(nx):
            YX[0][j] = np.eye(rx[0, j])
            YX[d][j] = np.eye(rx[d, j])
            ZX[0][j] = np.eye(rx[0, j])
            ZX[d][j] = np.eye(rx[d, j])

    d = len(n)
    tol_local = tol / np.sqrt(d)

    # Some place to store global indices
    Jy = [None] * (d + 1)
    Jz = [None] * (d + 1)

    # Choose where we start warm-up iteration
    dir = -dir
    istart = d
    if dir > 0:
        istart = 1
    iprev = (1 - dir) / 2 # where is the previous block relative to i
    inext = (1 + dir) / 2 # where is the next block relative to i

    # Find out what our initial guess is
    if isinstance(y, (int, float)):
        # Random samples for the initial indices
        if isinstance(y, (int, float)): # Matlab isscalar
            nq = y
            ind = np.random.rand(nq, d) 
            ind = ind * (n[:, None] - 1)
            ind = np.round(ind) + 1  # Round elements of the array and add 1
        else:
            ind = y
            nq = ind.shape[0]  # Get the number of rows in ind

        # Check if X is not empty
        if X:
            # Interface matrices
            for j in range(nx):
                YX[istart + iprev][j] = np.tile(YX[istart + iprev][j], (1, nq, 1))

            # Sample X at initial indices
            i = istart
            while i != (d - istart + 1):
                YX[i + inext] = sample_others_indep(YX[i + iprev], X[i], ind[:, i], dir)
                i = i + dir

            for j in range(nx):
                YX[istart + iprev][j] = YX[istart + iprev][j][:, 0, :]

        # Check if ifun and soughts are not empty
        if ifun or soughts:
            # Store the indices
            for i in range(2, d + 1):  # Python uses 0-based indexing, range goes up to but does not include the stop value
                if dir > 0:
                    Jy[i] = ind[:, :i - 1]  # Slicing in Python is similar to MATLAB, but note the 0-based indexing
                else:
                    Jy[i] = ind[:, i - 1:d]  # Adjusted indexing for Python

        # Initialize y
        y = [None] * d
        ry = [1] + [nq] * (d - 1) + [1]

    elif isinstance(y, list):
        ry = y.r # Attribute r of tt_tensor y
        y = core2cell(y)  # need core2cell

        i = istart
        while i != (d - istart + 1):
            # QR decomposition and index selection
            y[i - iprev], y[i + inext], ry[i + inext], ind = qrmaxvol_block(y[i - iprev], y[i + inext], dir, [])

            # Store indices if we have ifun or soughts
            if ifun or soughts:
                if dir > 0:
                    Jy[i + 1] = indexmerge(Jy[i], np.arange(1, n[i] + 1))
                else:
                    Jy[i] = indexmerge(np.arange(1, n[i] + 1), Jy[i + 1])
                Jy[i + inext] = Jy[i + inext][ind, :]

            # Sample X for ffun
            if X:
                YX[i + inext, :] = sample_others_nested(YX[i + iprev, :], X[i, :], ind, ry[i + iprev], dir)

            i += dir

        ry[0] = 1
        ry[d] = 1 # those might be different if a block tensor is given
        
    else:
        Jy = y

        # Initialize y
        y = [None] * d
        ry = np.ones(d + 1, dtype=int)
        i = istart

        while i != (d - istart + 1):
            ry[i + inext] = Jy[i + inext].shape[0]

            # Sample X for ffun
            if X:
                # Find local indices
                if dir > 0:
                    J = indexmerge(np.arange(1, n[i] + 1), Jy[i + 1])
                else:
                    J = indexmerge(Jy[i], np.arange(1, n[i] + 1))

                ind = np.zeros(ry[i + inext], dtype=int)
                for j in range(ry[i + inext]):
                    indj = np.where((np.tile(Jy[i + inext][j, :], (n[i] * ry[i + iprev], 1)) == J).all(axis=1))[0]
                    ind[j] = indj[int(np.round(np.random.rand() * (len(indj) - 1)))]
                
                YX[i + inext, :] = sample_others_nested(YX[i + iprev, :], X[i, :], ind, ry[i + iprev], dir)

            i += dir

    # Generate projections to residual Z
    if kickrank > 0:
        # Initialise rank of Z
        if abs(kickrank - round(kickrank)) < 1e-8:
            # kickrank is integer, this is the rank of z
            rz = kickrank
        else:
            # kickrank is a fraction of the rank of y
            rz = np.ceil(kickrank * max(ry)).astype(int)

        ind = np.random.rand(rz, d)
        ind = ind * (n[:, None] - 1)  # Assuming n is a numpy array
        ind = np.round(ind) + 1  # should be >=1 and <=n now

        # projection of solution Y
        ZY = [None] * (d + 1)
        ZY[0] = 1
        ZY[-1] = 1

        # If y is already here
        if not any([el is None for el in y]):  # Assuming y is a list of arrays or None
            ZY[istart + iprev] = np.ones((1, rz))
            i = istart
            while i != (d - istart + 1):
                ZY[i + inext] = sample_others_indep(ZY[i + iprev], y[i], ind[:, i], dir)
                i += dir
            ZY[istart + iprev] = 1
        else:
            # Otherwise populate with some random
            for i in range(d, 1, -1):
                ZY[i] = np.random.randn(ry[i], rz)
                if dir > 0:
                    ZY[i] = ZY[i].T

        if X:
            # Interface matrices
            for j in range(nx):
                ZX[istart + iprev, j] = np.tile(ZX[istart + iprev, j].reshape((ZX[istart + iprev, j].shape[0], 1, -1)), (1, rz, 1))
            # Sample x from the right
            i = istart
            while i != (d - istart + 1):
                ZX[i + inext, :] = sample_others_indep(ZX[i + iprev, :], X[i, :], ind[:, i], dir)
                i += dir
            for j in range(nx):
                ZX[istart + iprev, j] = ZX[istart + iprev, j][:, 0, :].reshape((ZX[istart + iprev, j].shape[0], -1))

        if ifun or soughts:
            # Store the indices
            Jz = [None] * (d + 1)
            for i in range(1, d):
                if dir > 0:
                    Jz[i] = ind[:, :i]
                else:
                    Jz[i] = ind[:, i:d]

        rz = [1] + [rz] * (d - 1) + [1]
    
    # Start the computation loop
    swp = 1
    dir = -dir
    istart = d - istart + 1
    i = istart
    iprev = (1 - dir) // 2 # seems like we need a integer here, do double check when testing
    inext = (1 + dir) // 2
    last_swp = 0
    max_dx = 0
    fevalcnt = 0
    ievalcnt = 0

    while swp <= nswp:
        if swp == 1 or i != istart:
            # Evaluate the new core
            cry, ievalcnt, fevalcnt = evaluate_fun(i, Jy, Jy, n, ifun, ffun, X, rx, YX, YX, vec, ievalcnt, fevalcnt)
        else:
            cry = y[i].reshape(-1, b)

        if soughts:
            Jy[i, 1] = indexmerge(Jy[i], np.arange(1, n[i] + 1), Jy[i + 1])

        # Block size
        if swp == 1 and i == istart:
            b = cry.shape[1]
            # Stat quantities
            statvals = np.zeros((len(soughts), 1, b))
            statind = np.zeros((len(soughts), d, b))

        # Check if the user function is sane
        if cry.size != ry[i] * n[i] * ry[i + 1] * b:
            raise ValueError(f'{ry[i] * n[i] * ry[i + 1]} elements requested, but {cry.size // b} values received. Check your function or use vec=false')
        
        # Stat outputs
        for j in range(len(soughts)):
            case = soughts[j].lower()
            
            if case == 'lm':
                # Compute the maximum absolute value and its index
                val, ind = np.max(np.abs(cry)), np.argmax(np.abs(cry))
                to_update = val > np.abs(statvals[j, :])
            elif case == 'sm':
                val, ind = np.min(np.abs(cry)), np.argmin(np.abs(cry))
                to_update = val < np.abs(statvals[j, :])
            elif case == 'lr':
                val, ind = np.max(np.real(cry)), np.argmax(np.real(cry))
                to_update = val > np.real(statvals[j, :])
            elif case == 'sr':
                val, ind = np.min(np.real(cry)), np.argmin(np.real(cry))
                to_update = val < np.real(statvals[j, :])
            elif case == 'li':
                val, ind = np.max(np.imag(cry)), np.argmax(np.imag(cry))
                to_update = val > np.imag(statvals[j, :])
            elif case == 'si':
                val, ind = np.min(np.imag(cry)), np.argmin(np.imag(cry))
                to_update = val < np.imag(statvals[j, :])
            
            # If any new values are better, update them
            if (swp == 1) and (i == istart):  # Of course, in the first step update all
                to_update = np.ones(b, dtype=bool)
            
            if np.any(to_update):
                statvals[j, 0, to_update] = np.diag(cry[ind[to_update], to_update])
                ind = np.unravel_index(ind[to_update], (ry[i], n[i], ry[i + 1]))
                
                if i > 1:
                    statind[j, 0:i, to_update] = np.transpose(Jy[i][ind[to_update, 0], :])
                
                statind[j, i, to_update] = ind[to_update, 1]
                
                if i < d:
                    statind[j, i + 1:d, to_update] = np.transpose(Jy[i + 1][ind[to_update, 2], :])

        # Estimate the error -- now in C-norm
        if y[i] is None or (swp == 1 and i == istart and b > 1):
            y[i] = np.zeros((ry[i] * n[i] * ry[i + 1] * b, 1))
        
        y[i] = y[i].reshape((ry[i] * n[i] * ry[i + 1] * b, 1))
        cry = cry.reshape((ry[i] * n[i] * ry[i + 1] * b, 1))
        dx = np.max(np.abs(cry - y[i])) / np.max(np.abs(cry))
        max_dx = max(max_dx, dx)

        # Switch to the next block
        cry = cry.reshape((ry[i], n[i], ry[i + 1], b))
        y[i] = cry


        if i != (d - istart + 1):
            # We are at an intermediate block in the current sweep
            # Truncation
            y[i - iprev], y[i + inext], ry[i + inext], cry = truncate_block(y[i - iprev], y[i + inext], tol_local, dir)
            crs = []

            # Enrichment
            if kickrank > 0:
                crz, crs = project_solution_to_residual(cry, dir, ZY[i], ZY[i + 1])
                # Evaluate the Z-core
                if dir > 0:
                    crzex, ievalcnt, fevalcnt = evaluate_fun(i, Jy, Jz, n, ifun, ffun, X, rx, YX, ZX, vec, ievalcnt, fevalcnt)
                    crzex = crzex.reshape((ry[i], n[i], -1, b))
                else:
                    crzex, ievalcnt, fevalcnt = evaluate_fun(i, Jz, Jy, n, ifun, ffun, X, rx, ZX, YX, vec, ievalcnt, fevalcnt)
                    crzex = crzex.reshape((-1, n[i], ry[i + 1], b))

                crs = crzex - crs
                crs = truncate_residual(crs, dir, kickrank, tol_local, ry[i + inext], False)

            # Enrich, QR, maxvol
            y[i - iprev], y[i + inext], ry[i + inext], ind = qrmaxvol_block(y[i - iprev], y[i + inext], dir, crs)

            # Restrict left indices/matrices
            if ifun or soughts:
                if dir > 0:
                    Jy[i + inext] = indexmerge(Jy[i], np.arange(1, n[i] + 1), Jy[i + 1])
                else:
                    Jy[i + inext] = indexmerge(np.arange(1, n[i] + 1), Jy[i + 1])
                Jy[i + inext] = Jy[i + inext][ind, :]

            if X:
                YX[i + inext, :] = sample_others_nested(YX[i + iprev, :], X[i, :], ind, ry[i + iprev], dir)

            # Update the residual itself
            if kickrank > 0:
                # Evaluate the Z-core
                crzex, ievalcnt, fevalcnt = evaluate_fun(i, Jz, Jz, n, ifun, ffun, X, rx, ZX, ZX, vec, ievalcnt, fevalcnt)
                crzex = crzex.reshape((rz[i], n[i], -1, b))
                crz = crzex - crz
                crz = truncate_residual(crz, dir, kickrank, tol_local, ry[i + inext], True)
                rz[i + inext] = crz.shape[1]  # Assuming rz is a list or array
                ind = maxvol2(crz)  # maxvol2 function needs to be defined 
                # Restrict left indices/matrices
                if ifun or soughts:  # Check if ifun or soughts are not empty
                    if dir > 0:
                        Jz[i + inext] = indexmerge(Jz[i], np.arange(1, n[i] + 1))
                    else:
                        Jz[i + inext] = indexmerge(np.arange(1, n[i] + 1), Jz[i + 1])
                    Jz[i + inext] = Jz[i + inext][ind, :]

                if X:
                    ZX[i + inext, :] = sample_others_nested(ZX[i + iprev, :], X[i, :], ind, rz[i + iprev], dir)
            
                ZY[i + inext] = sample_others_nested(ZY[i + iprev], y[i], ind, rz[i + iprev], dir)
        else:
            # Terminal core of the current sweep
            y[i] = cry

        if verb > 1:
            print(f"\t-amen_cross_s- swp={swp}, i={i}, dx={dx:.3e}, ranks=[{ry[i]},{ry[i + 1]}], n={n[i]}")

        i += dir

        # Change direction, check for exit
        if i == (d - istart + 1 + dir):
            if verb > 0:
                print(f"=amen_cross_s= swp={swp}, max_dx={max_dx:.3e}, max_rank={max(ry)}, max_n={max(n)}, cum#ievals={ievalcnt}, cum#fevals={fevalcnt}")

            if max_dx < tol_exit:
                last_swp += 1

            if (last_swp > stop_sweep or swp >= nswp) and (dir == exitdir or exitdir == 0):
                break

            dir = -dir
            istart = d - istart + 1
            iprev = (1 - dir) // 2  # where is the previous block relative to i
            inext = (1 + dir) // 2  # where is the next block relative to i
            swp += 1
            max_dx = 0
            i += dir

    if dir < 0:
        y[0] = y[0].reshape((n[0], ry[1], b))
        y[0] = np.transpose(y[0], (2, 0, 1))
    else:
        y[-1] = y[-1].reshape((ry[-1], n[-1], b))

    y = cell2core(tt_tensor, y) # need core2cell to be defined

    # Combine evaluation counts
    evalcnt = [ievalcnt, fevalcnt]

    return y, statvals, statind, Jy, evalcnt