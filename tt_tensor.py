import numpy as np
from scipy.linalg import svd

def my_chop2(sv, eps):
    """
    Truncation by absolute precision in Frobenius norm.
    Finds the minimal possible r such that sqrt(sum(sv[r:]**2)) < eps.
    """
    if np.linalg.norm(sv) == 0:
        return 1

    if eps <= 0:
        return len(sv)

    sv0 = np.cumsum(sv[::-1]**2)
    ff = np.where(sv0 < eps**2)[0]

    if ff.size == 0:
        return len(sv)
    else:
        return len(sv) - ff[-1] - 1
    
class tt_tensor:
    def __init__(self, array=None, eps=1e-14):
        if array is None:
            self.d = 0
            self.r = 0
            self.n = 0
            self.core = 0    # empty tensor
            self.ps = 0
            self.over = 0    # estimate of the rank over the optimal
        else:
            self.convert_from_full(array, eps)

    def convert_from_full(self, array, eps):
        n = array.shape
        d = len(n)
        r = np.ones(d + 1, dtype=int)

        if len(n) == 1 or (len(n) == 2 and min(n) == 1) or issparse(array): #currently no output if issparse(array) = False
            r = np.array([r[0], r[d]])
            d = 1
            n = np.prod(n)
            core = array.ravel()
            ps = np.cumsum(np.hstack(([0], n * r[:d] * r[1:d + 1] - 1)))
            self.d = d
            self.n = n
            self.r = r
            self.ps = ps
            self.core = core
            self.over = 0
            return

        c = array
        core = []
        pos = 1
        ep = eps / np.sqrt(d - 1)

        for i in range(d - 1):
            m = n[i] * r[i]
            c = c.reshape((m, -1))
            u, s, v = np.linalg.svd(c, full_matrices=False)
            r1 = my_chop2(s, ep * np.linalg.norm(s))
            u = u[:, :r1]
            s = s[:r1]
            r[i + 1] = r1
            core.append(u.reshape(r[i], n[i], r[i + 1]))
            v = v[:r1, :]
            v = np.dot(np.diag(s), v)
            c = v.T
            pos += r[i] * n[i] * r[i + 1]

        core.append(c.reshape(r[d - 1], n[d - 1], r[d]))
        self.core = np.concatenate(core, axis=None)
        self.ps = np.cumsum(np.hstack(([0], - 1 + n * r[:d] * r[1:d + 1] - 1))) 
        self.d = d
        self.n = n
        self.r = r
        self.over = 0