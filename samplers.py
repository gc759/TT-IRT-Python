import numpy as np
import scipy.special as sp

def randref(reference, sizes=None):
    """
    Samples uniform or truncated normal pseudorandom numbers
    Input parameters:
    - reference (str): Specifies the reference density. Options are:
        - 'UNIform' (or any string starting with 'u') for uniform distribution.
        - 'Normal' or 'Normal S' for truncated normal distribution,
          where S is the number of sigmas defining the [-S,S] support
          of the reference variables. Defaults to S=4 if not specified.
    - sizes: arbitrary combination of array sizes compatible with np.random.rand.
             If not provided, you can pass an array of numbers in [0,1]
             (e.g. QMC lattice points) that will be transformed to reference.

    Returns:
    - y: Array of samples from the specified distribution.

    Raises:
    - ValueError: If sizes is None and the second argument is not an array of samples.

    Examples:
    # Generate a 3x3 array of uniform random numbers
    rand_nums_uniform = randref('uniform', (3, 3))

    # Generate a 3x3 array of truncated normal random numbers with default sigma
    rand_nums_truncated_normal = randref('Normal', (3, 3))

    # Generate a 3x3 array of truncated normal random numbers with sigma = 3
    rand_nums_truncated_normal_sigma3 = randref('Normal 3', (3, 3))

    # Transform an existing array of uniform numbers in [0, 1] to truncated normal distribution with sigma = 3
    uniform_numbers = np.random.rand(10)
    transformed_numbers_sigma3 = randref('Normal 3', uniform_numbers)
    """
    if sizes is not None:
        # Sizes are provided, sample corresponding Uniform
        y = np.random.rand(*sizes)
    else:
        # Sizes are not provided, expecting y as the second argument
        raise ValueError("Sizes must be provided if the second argument is not an array of samples.")

    if reference.lower()[0] != 'u':
        # Truncated normal
        sigma_str = ''.join(filter(str.isdigit, reference.lower() + '.'))
        sigma = float(sigma_str) if sigma_str else 4
        
        # Multiply erf by this to get the truncated CDF between 0 and 1
        cdf_ifactor = sp.erf(sigma / np.sqrt(2)) / 0.5
        
        y = sp.erfinv((y - 0.5) * cdf_ifactor) * np.sqrt(2)

    return y


def mcmc_prune(y, lFex, lFapp):
    """
    Performs a simple MCMC rejection loop with independent proposals.

    Parameters:
    - y (np.ndarray): Proposal samples.
    - lFex (np.ndarray): Array containing [log(exact density), Quantity of Interest] evaluated at y.
    - lFapp (np.ndarray): Log of proposal density evaluated at y.

    Returns:
    - y (np.ndarray): Pruned samples.
    - lFex (np.ndarray): Exact data at new y.
    - lFapp (np.ndarray): Log of proposal density at new y.
    - num_of_rejects (int): Total number of rejections.
    - rej_distribution (np.ndarray): Unnormalized distribution function of the number of
      consecutive rejections, i.e., rej_distribution[L] ~ Prob(lag==L).

    Examples:
    # Example usage:
    y = np.random.rand(100, 1)
    lFex = np.random.rand(100, 2)
    lFapp = np.random.rand(100)
    y, lFex, lFapp, num_of_rejects, rej_distribution = mcmc_prune(y, lFex, lFapp)
    """
    rej_distribution = np.zeros(1)

    M = len(lFapp)
    num_of_rejects = 0
    rej_seq = 0

    for i in range(M - 1):
        alpha = lFex[i + 1, 0] - lFex[i, 0] - lFapp[i + 1] + lFapp[i]
        alpha = np.exp(alpha)

        if alpha < np.random.rand():
            # Reject: Copy the i-th data to (i+1)
            y[i + 1, :] = y[i, :]
            lFapp[i + 1] = lFapp[i]
            lFex[i + 1, :] = lFex[i, :]
            num_of_rejects += 1
            rej_seq += 1
        elif rej_seq > 0:
            # Accept: Save the previous rej_seq and reset it
            if len(rej_distribution) >= rej_seq:
                rej_distribution[rej_seq - 1] += 1
            else:
                rej_distribution = np.pad(rej_distribution, (0, rej_seq - len(rej_distribution)), 'constant')
                rej_distribution[rej_seq - 1] = 1
            rej_seq = 0

    print(f'mcmc_prune completed with {num_of_rejects} rejections ({num_of_rejects / M * 100:.2f}%)')

    return y, lFex, lFapp, num_of_rejects, rej_distribution


def essinv(lFex, lFapp):
    """
    Calculates the normalized Inverse Effective Sample Size (ESS), 
    an analog of IACT (Integrated Autocorrelation Time) for Importance Weighting.
    Also provides an estimate to 1 + chi-squared divergence.

    Parameters:
    - lFex (np.ndarray): Log of exact density values.
    - lFapp (np.ndarray): Log of sampling density values.

    Returns:
    - tau (float): Normalized Inverse Effective Sample Size (N/ESS).

    Example:
    # Example usage:
    lFex = np.log(np.random.rand(10))
    lFapp = np.log(np.random.rand(10))
    tau = essinv(lFex, lFapp)
    """
    dF = lFex - lFapp
    dF = dF - np.max(dF)
    tau = len(lFapp) * np.sum(np.exp(dF * 2)) / np.sum(np.exp(dF))**2
    return tau

def hellinger(lFex, lFapp):
    """
    Approximates the Hellinger distance from samples.

    The Hellinger distance is calculated using the formula:
    2*H^2 = E_{Fapp}[ \sqrt{(Fex/Zex)/Fapp} - 1 ]^2

    Parameters:
    - lFex (np.ndarray): Log of exact density Fex, can be unnormalized.
    - lFapp (np.ndarray): Log of sampling density Fapp, must be normalized.

    Returns:
    - H (float): Approximated Hellinger distance.

    Example:
    # Example usage:
    lFex = np.log(np.random.rand(10))
    lFapp = np.log(np.random.rand(10))
    H = hellinger(lFex, lFapp)
    """
    dF = lFex - lFapp
    dF = dF - np.max(dF)
    lZex = np.log(np.mean(np.exp(dF)))  # Up to +max(dF) which cancels below anyway
    H = np.mean((np.exp(0.5 * (dF - lZex)) - 1)**2)
    H = np.sqrt(H / 2)
    return H


def tt_dirt_sample(IRTstruct, q, logpostfun=None, vec=True):
    """
    Samples a density represented by a Deep Inverse Rosenblatt Transform (DIRT).

    Parameters:
    - IRTstruct (dict): Structure from tt_dirt_approx with density ratios in TT.
    - q (np.ndarray): An M x d array of seed points on [0,1]^d for uniform reference, or
                      on [-S,S]^d for truncated normal reference.
    - logpostfun (function, optional): Function of exact target log-density.
    - vec (bool, optional): Whether logpostfun can process vectorized x. Default is True.

    Returns:
    - z (np.ndarray): Transformed samples from q.
    - lFapp (np.ndarray): Log(pushforward density) (inverse Jacobian of z).
    - lFex (np.ndarray): Samples of log(exact density) (if logpostfun was given).

    Example:
    # Example usage:
    IRTstruct = {
        'beta': np.array([0, 1]),
        'reference': 'uniform',
        'crossmethod': 'spline',
        'F': [np.array([1]), np.array([1])],  # Mock-up of density ratios in TT
        'F0': np.array([1])  # Mock-up of initial density ratio in TT
    }
    q = np.random.rand(10, 2)
    z, lFapp, lFex = tt_dirt_sample(IRTstruct, q, logpostfun=lambda x: -np.sum(x**2, axis=1))
    """

    nlvl = len(IRTstruct['beta']) - 1
    lFapp = 0
    z = q

    if IRTstruct['reference'][0].lower() != 'u':
        sigma = extract_sigma(IRTstruct['reference'])
        cdf_factor = 0.5 / sp.erf(sigma / np.sqrt(2))

    # Sample in reversed order
    for i in range(nlvl, 0, -1):
        if IRTstruct['reference'][0].lower() != 'u':
            z = sp.erf(z / np.sqrt(2)) * cdf_factor + 0.5  # Transform TNormal -> uniform

        # Transform the current level
        z, dlFapp = transform_level(IRTstruct['F'][i-1], z, IRTstruct)
        lFapp += dlFapp  # Add log(Jacobian)

    # Sample Level 0
    z, dlFapp = transform_level(IRTstruct['F0'], z, IRTstruct)
    lFapp += dlFapp

    # Exact density
    lFex = None
    if logpostfun is not None:
        lFex = logpostfun_vec(z, logpostfun, vec)

    return z, lFapp, lFex

def extract_sigma(reference):
    """Extracts sigma value from the reference string."""
    sigma_str = ''.join(filter(str.isdigit, reference.lower() + '.'))
    return float(sigma_str) if sigma_str else 4

def transform_level(F, z, IRTstruct):
    """Transforms the current level, placeholder for actual transformation logic."""
    # Placeholder for the transformation logic based on IRTstruct
    # Returning dummy values for demonstration
    return z, np.zeros_like(z[:, 0])

def logpostfun_vec(x, logpostfun, vec):
    """Wrapper for vectorized or non-vectorized user function."""
    if vec:
        return logpostfun(x)
    else:
        return np.array([logpostfun(i) for i in x])