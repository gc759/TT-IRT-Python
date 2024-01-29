import numpy as np
from amen_cross_s import *
import matplotlib.pyplot as plt
from matplotlib import cm

import re

from samplers import *
from tt_tensor import *
from tt_toolbox import *

# to be defined
# full
# int_block
# round_function
# Legendre
# Fourier
# ftt.m and oned class

def logpostfun_vec(x, beta_min, beta_max, logpostfun, vec):
    if vec:
        y = logpostfun(x, beta_min, beta_max)
    else:
        y = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            y[i] = logpostfun(x[i, :], beta_min, beta_max)
    return y
    
def dualbetafun(x, IRTstruct, logpostfun, beta, beta_prev, lFshift, vec, reference, IRTdenom):
    # Sample existing DIRT
    z, lFapp = tt_dirt_sample(IRTstruct, x)
    
    if IRTdenom:
        # Compute ratio with the IRT density in the denominator
        beta_prev = 0

    # Exact log-ratio
    F = logpostfun_vec(z, beta_prev, beta, logpostfun, vec)
    F = F - lFshift  # Remove baseline to prevent overflow

    if IRTdenom:
        F = F - lFapp

    if reference[0] != 'u':
        # Remove the reference log-density
        F = F - np.sum(x**2, axis=1) / 2

    # SQRT(Density)
    F = np.exp(F * 0.5)

    return F
    
def tt_dirt_approx(x0, logpostfun, beta, nq=None, stoptol=0.4, trunctol=0,
                   crossmethod='amen_cross_s', y0=1, kickrank=4, nswp=4,
                   vec=True, boundary=False, testsamples=1e4, recompute=50,
                   reference='uni', IRTdenom=False, plotdiag=True, interpolation='spline',
                   IRTstruct=None):
    '''
    x0 (list of np.ndarray or oned classes):
        list of NumPy arrays v1, ... , vd of initial grid vectors on level 0, or 'oned' classes from ftt.m
    '''
    
    nlvl = len(beta) - 1  # number of increment levels
    d = len(x0)  # number of variables
    
    # Allow vector-valued cross parameters to fine-tune different levels
    if len(nswp) == 1:
        nswp = np.tile(nswp, nlvl+1)
    if len(kickrank) == 1:
        kickrank = np.tile(kickrank, nlvl+1)
    if len(stoptol) == 1:
        stoptol = np.tile(stoptol, nlvl+1)
    if len(trunctol) == 1:
        trunctol = np.tile(trunctol, nlvl+1)
    if len(IRTdenom) == 1:
        IRTdenom = np.tile(IRTdenom, nlvl+1)
    if y0.shape[0] == 1:
        y0 = np.tile(y0, (d+1, 1))  # we may need to fine-tune ranks over dimensions and levels
    if y0.shape[1] == 1:
        y0 = np.tile(y0, (1, nlvl+1))

    # Override boundary to True for Fourier interpolation if needed
    if interpolation[0] != 's' and not boundary:
        boundary = True
        print("Warning: Overriding boundary->True for Fourier interpolation")

    # Zero level needs special treatment, as the box for X can be arbitrary
    if isinstance(x0[0], oned): # to be defined, oned is a class from ftt.m 
        # # This is for FTT, check if x0 is a cell of oned polys
        # crossmethod = 'build_ftt'
        # nq = [p.order for p in x0]
        pass
    else:
        # This is for gridpoints and TT-Toolbox
        if not nq:
            # nq is an empty list
            nq = [len(xi) for xi in x0]
        if boundary:
            X = tt_meshgrid_vert([tt_tensor(xi) for xi in x0])
        else:
            X = tt_meshgrid_vert([tt_tensor(xi[1:-1]) for xi in x0])
    if len(nq) == 1:
        nq = [nq[0]] * d # Assuming x0's length is the dimension d
        
    # Storage for densities and functions
    F = [None] * nlvl

    if not IRTstruct:
        # Initialize empty IRT structure
        ilvl = 0
        IRTstruct = {}
        IRTstruct['x0'] = x0
        IRTstruct['beta'] = beta[0]
        IRTstruct['reference'] = 'reference'
        IRTstruct['crossmethod'] = 'crossmethod'
        IRTstruct['interpolation'] = 'interpolation'
        # Initial guess
        Fprev = np.max(y0[:, min(1, y0.shape[1] - 1)])
    else:
        ilvl = len(IRTstruct['beta'])
        if ilvl > 1:
            F[:ilvl - 1] = IRTstruct['F'][:ilvl - 1]
        beta[:ilvl] = IRTstruct['beta']
        lFshift = IRTstruct['lFshift']
        Fprev = IRTstruct['Fprev']
    
    if ilvl == 0:
        print(f"Approximating level 0, for beta={beta[0]}")
        if crossmethod == 'amen_cross_s':
            F0, _, _, _, evalcnt1 = amen_cross_s(X, lambda x: np.exp(logpostfun_vec(x, 0, beta[0], logpostfun, vec) * 0.5),
                                                 trunctol[0], tol_exit=stoptol[0], y0=max(y0[:, 0]), kickrank=kickrank[0],
                                                 nswp=nswp[0], verb=1)
        elif crossmethod == 'greedy2_cross': # from TT-Toolbox
            pass # to be defined later
        elif crossmethod == 'build_ftt': # from ftt.m
            pass # to be defined later

        IRTstruct['evalcnt'][0] = np.sum(evalcnt1)

        if isinstance(F0, tt_tensor): # check if it's tensor class
            if plotdiag:
                if plotdiag:
                    # Draw 1D marginals
                    Fdiag = np.zeros((max(F0.n), d))
                    Fdiag[:F0.n[0], 0] = full(dot(tt_ones(F0.n[1:d]), F0, 2, d))
                    for i in range(1, d-1):
                        tensor_1 = tt_ones(F0.n[i:d])
                        tensor_2 = dot(tt_ones(F0.n[0:i-1]), F0, 1, i-1)
                        Fdiag[:F0.n[i], i] = full(dot(tensor_1, tensor_2, 2, d-i+1))
                        # Anything related to indecies needs to be checked.

                # Desintegrate tt_tensor into cell array for faster referencing
                F0 = core2cell(F0)

        if isinstance(F0, FTT) and plotdiag:  # FTT is a class, need to be defined
            # Draw 1D marginals
            Fdiag = []
            for i in range(d):
                Xplot = x0[i].nodes
                Fdiag.append(eval(int_block(F0, list(range(1, i)) + list(range(i + 1, d))), Xplot))

            plt.figure(1)
            for fd in Fdiag:
                plt.plot(fd)
            plt.legend()
            plt.title('1D marginal sqrt(densities)')

            # Draw 2D marginal
            plt.figure(2)
            Xplot, Yplot = np.meshgrid(x0[0].nodes, x0[1].nodes)
            if d == 2:
                Z = np.reshape(eval(F0, np.column_stack([Xplot.ravel(), Yplot.ravel()])), Xplot.shape)
            else:
                Z = np.reshape(eval(int_block(F0, list(range(3, d))), np.column_stack([Xplot.ravel(), Yplot.ravel()])), Xplot.shape)

            plt.pcolormesh(Xplot, Yplot, Z, shading='auto')
            plt.colorbar()
            plt.title('2D x_1 x_2 marginal')
            plt.show()

        # Populate the DIRT structure with the zeroth level
        IRTstruct['F0'] = F0
        IRTstruct['Fprev'] = np.max(y0[:, min(1, y0.shape[1] - 1)])

        lFshift = 0
        if testsamples > 0:
            # Test approximation
            y = randref(reference, (min(sum(evalcnt1), testsamples), d))  # Need to define randref, from Matlab sampler folder
            y, lFapp, lFex = tt_dirt_sample(IRTstruct, y, lambda x: logpostfun(x, 0, beta[0]), vec) # Need to define, from Matlab sampler folder
            y2, _, _, num_of_rejects = mcmc_prune(y, lFex, lFapp) # Need to define, from Matlab sampler folder
            num_of_rejects = num_of_rejects * 100 / y.shape[0]
            tau = essinv(lFex, lFapp)  # Need to define, from Matlab sampler folder
            print(f'N/ESS = {tau}\n\n')
            
            if plotdiag:
                plt.figure(3)
                plt.plot(y2)
                plt.title(f'Chain: #rejects = {num_of_rejects}%, N/ESS = {tau}')
                plt.show()

            IRTstruct['evalcnt'][0] = IRTstruct.get('evalcnt', [0])[0] + y.shape[0]
            lFshift = np.max(lFex)  # We will subtract this to prevent overflows
            if IRTdenom[0]:
                lFshift -= np.max(lFapp)
            
            IRTstruct['lFshift'] = lFshift

        ilvl += 1

    # Parsing domain size for truncated normal
    # extract a numerical value from a string
    # Line 272 to 281 in Matlab file

    if reference[0] != 'u':
        # Extract numbers and dot from the string reference
        sigma_str = re.findall(r'[\d.]+', reference)

        # Convert the first match to float if it exists, else default to 4
        if sigma_str:
            sigma = float(sigma_str[0])
        else:
            sigma = 4

        print(f'Using normal reference on [{-sigma}, {sigma}]')

    # Set up 1D ansatz for other levels
    if crossmethod == 'build_ftt':
        if reference[0] == 'u':
            x = [Legendre(n, [0, 1]) for n in nq]
        else:
            x = [Fourier(n, [-sigma, sigma]) for n in nq]
    else:
        if reference[0] == 'u':
            x = [0.5 * (np.cos(np.pi * np.arange(n - 1, -1, -1) / (n - 1)) + 1) for n in nq]
        else:
            if interpolation[0] == 's':
                x = [(np.arange(0, 1, 1 / (n - 1)) * 2 * sigma - sigma) for n in nq]
            else:
                x = [(np.arange(1, n + 1) * (2 * sigma / n) - sigma) for n in [round(ni / 2) * 2 for ni in nq]]

        if boundary:
            X = tt_meshgrid_vert([tt_tensor(xi) for xi in x])
        else:
            X = tt_meshgrid_vert([tt_tensor(xi[1:-1]) for xi in x])

        x = np.hstack(x)

    IRTstruct['x'] = x

    recompute_count = 0  # Count the number of unsuccessful TT-Crosses

    while ilvl <= nlvl:
        print(f'Approximating level {ilvl}, for beta={beta[ilvl + 1]}')

        if crossmethod == 'amen_cross_s':
            F[ilvl], _, _, _, evalcnt1 = amen_cross_s(X, dualbetafun, trunctol[ilvl + 1],
                                                      tol_exit=stoptol[ilvl + 1], y0=Fprev, kickrank=kickrank[ilvl + 1], nswp=nswp[ilvl + 1], verb=1)

        elif crossmethod == 'greedy2_cross':
            pass # replace later

        elif crossmethod == 'build_ftt':
            pass  # replace later

        # record the number of evaluations
        IRTstruct['evalcnt'][ilvl + 1] = IRTstruct.get('evalcnt', [0])[ilvl + 1] + sum(evalcnt1)

        if isinstance(F[ilvl], tt_tensor):
            if plotdiag:
                # Draw 1D marginals
                Fdiag = np.zeros((max(F[ilvl].n), d))
                Fdiag[:F[ilvl].n[0], 0] = full(dot(tt_ones(F[ilvl].n[1:d]), F[ilvl], 2, d))

                for i in range(1, d-1):
                    tensor_1 = tt_ones(F{ilvl}.n(i:d))
                    tensor_2 = dot(tt_ones(F{ilvl}.n(0:i-1)), F{ilvl}, 1, i-1)
                    Fdiag[:F[ilvl].n[i], i] = full(dot(tensor_1, tensor_2, 2, d-i+1))

                Fdiag[:F[ilvl].n[-1], -1] = full(dot(tt_ones(F[ilvl].n[0:d-1]), F[ilvl], 1, d-1))

                plt.figure(1)
                plt.plot(Fdiag)
                plt.legend()
                plt.title('1D marginal sqrt(densities)')

                # Draw 2D marginal
                plt.figure(2)
                if d == 2:
                    plt.imshow(full(F[ilvl], F[ilvl].n[:2]), cmap='viridis', interpolation='none')
                else:
                    contraction = full(dot(tt_ones(F[ilvl].n[2:])/np.prod(F[ilvl].n[2:]), F[ilvl], 3, d), F[ilvl].n[:2])
                    plt.imshow(contraction, cmap='viridis', interpolation='none')
                
                plt.colorbar()
                plt.title('2D u_1 u_2 marginal')
                plt.show()

            # Initial guess for the next step
            if y0.shape[1] < (ilvl + 2):  # Check if y0 has enough columns
                # Continue with the last prescribed TT rank gracefully
                y0[:, ilvl + 1] = y0[:, -1]

            # Initial guess with rank y0
            Fprev = round_function(F[ilvl], 0, y0[:, ilvl + 1])
            IRTstruct['Fprev'] = Fprev

            # Disintegrate into cells for faster sampling
            F[ilvl] = core2cell(F[ilvl])

        if isinstance(F[ilvl], FTT) and plotdiag:
            # Draw 1D marginals
            Fdiag = np.zeros((max(len(xi.nodes) for xi in x), d))
            for i in range(d):
                Xplot = x[i].nodes.T
                Fdiag[:len(Xplot), i] = eval(int_block(F[ilvl], list(range(1, i)) + list(range(i + 1, d))), Xplot)

            plt.figure(1)
            plt.plot(Fdiag)
            plt.legend()
            plt.title('1D marginal sqrt(densities)')

            # Draw 2D marginal
            plt.figure(2)
            Xplot, Yplot = np.meshgrid(x[0].nodes, x[1].nodes)
            if d == 2:
                Z = np.reshape(eval(F[ilvl], np.column_stack([Xplot.ravel(), Yplot.ravel()])), Xplot.shape)
            else:
                Z = np.reshape(eval(int_block(F[ilvl], list(range(3, d))), np.column_stack([Xplot.ravel(), Yplot.ravel()])), Xplot.shape)

            plt.pcolormesh(Xplot, Yplot, Z, shading='auto', cmap=cm.viridis)
            plt.colorbar()
            plt.title('2D u_1 u_2 marginal')
            plt.show()

        # Record the current DIRT stack
        IRTstruct['F'] = F[:ilvl + 1]
        IRTstruct['beta'] = beta[:ilvl + 2]

        if testsamples > 0:
            # Test approximation
            y = randref(reference, (min(sum(evalcnt1), testsamples), d))
            y, lFapp, lFex = tt_dirt_sample(IRTstruct, y, lambda x: logpostfun(x, 0, beta[ilvl + 1]), vec)
            y2, _, _, num_of_rejects = mcmc_prune(y, lFex, lFapp)
            num_of_rejects = num_of_rejects * 100 / len(y)
            tau = essinv(lFex, lFapp)
            hl = hellinger(lFex, lFapp)
            print(f'N/ESS = {tau}, Hellinger = {hl:.3e}')

            if plotdiag:
                plt.figure(3)
                plt.plot(y2)
                plt.title(f'Chain: #rejects = {num_of_rejects:.0f}%, N/ESS = {tau}, H = {hl:.3e}')
                plt.show()

            IRTstruct['evalcnt'][ilvl + 1] = IRTstruct.get('evalcnt', [0])[ilvl + 1] + len(y)

            if tau > recompute:
                ilvl -= 1
                recompute_count += 1
                if recompute_count > 4:
                    raise Exception(f'Too poor approximation at beta={beta[ilvl + 2]} after 5 attempts, giving up')
            else:
                if ilvl < nlvl:
                    if IRTdenom[ilvl + 1]:
                        lFshift = max(lFex) * beta[ilvl + 2] / beta[ilvl + 1] - max(lFapp)
                    else:
                        lFshift = max(lFex) * (beta[ilvl + 2] - beta[ilvl + 1]) / beta[ilvl + 1]
                    IRTstruct['lFshift'] = lFshift
                recompute_count = 0

        ilvl += 1
    
    return IRTstruct