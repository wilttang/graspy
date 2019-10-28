import numpy as np
from graspy.simulations import sample_edges
from graspy.utils import symmetrize, cartprod
import matplotlib.pyplot as plt
import copy
import warnings


def sample_corr(P, Rho, directed=False, loops=False):
    """
    Generate a correlated Erdos-Renyi graph G2 based on G1 graph
    with Bernoulli distribution.

    Both G1 and G2 are binary matrices. 
    If the value of the position of graph 1 is 1, 
    the probability of the same position of graph 2 to be 1 
    follows the Bernoulli distribution of
    (P[i][j])+Rho[i][j]*(1-P[i][j]).
    If the value of the position of graph 1 is 0, 
    the probability of the same position of graph 2 to be 1 
    follows the Bernoulli distribution of
    P1[i][j] = P[i][j]*(1-Rho[i][j]).

    Parameters
    ----------
    P: np.ndarray, shape (n_vertices, n_vertices)
        Matrix of probabilities (between 0 and 1) for a random graph
    Rho: np.ndarray, shape (n_vertices, n_vertices)
        Matrix whose elements are real numbers in the interval [0,1]
        Another probability matrix to definite the correlation between graph1 and graph2
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    References
    ----------
    .. [1] Vince Lyzinski, et al. "Seeded Graph Matching for Correlated Erd}os-Renyi Graphs", 
        Journal of Machine Learning Research 15, 2014
        
    Returns
    -------
    G2: ndarray (n_vertices, n_vertices)
        Binary matrix the same size as P representing a random graph,
        based on correlation matrix with P and Rho

    Examples
    --------
    >>> p = 0.5
    >>> rho = 0.3
    >>> n = 6
    >>> P = p * np.ones( (n,n) )
    >>> Rho = rho * np.ones((n,n))

    To sample a correlated graph based on P and Rho matrices:

    >>> sample_corr(P,Rho,directed = False, loops = False)
    array([[0. 1. 1. 1. 0. 0.]
           [1. 0. 1. 0. 0. 0.]
           [1. 1. 0. 1. 0. 0.]
           [1. 0. 1. 0. 1. 1.]
           [0. 0. 0. 1. 0. 0.]
           [0. 0. 0. 1. 0. 0.]])
    """
    if Rho is None:
        Rho = P
    if type(P) is not np.ndarray or type(Rho) is not np.ndarray:
        raise TypeError("P must be numpy.ndarray")
    if len(P.shape) != 2:
        raise ValueError("P must have dimension 2 (n_vertices, n_dimensions)")
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    if P.shape != Rho.shape:
        raise ValueError("Dimensions of matrices P and Rho must be the same")
    
    if not directed:
        # can cut down on sampling by ~half
        triu_inds = np.triu_indices(P.shape[0])
        samples = np.random.binomial(1, P[triu_inds])
        A = np.zeros_like(P)
        A[triu_inds] = samples
        A = symmetrize(A, method="triu")
    else:
        A = np.random.binomial(1, P)

    if loops:
        return A
    else:
        return A - np.diag(np.diag(A))

    # generate a Erdos-Renyi graph G1
    G1 = sample_edges(P, directed = False, loops = False) 
    origin_G1 = copy.deepcopy(G1)
    prob1 = origin_G1.sum()/(n*(n-1)) # calculate the probability of 1 shown in G1
    P1 = copy.deepcopy(P)
    Rho = copy.deepcopy(Rho)

    # renew the correlation between two graphs
    for i in range(n): 
        for j in range(n):
            if G1[i][j] == 1:
                P1[i][j] = P[i][j]+Rho[i][j]*(1-P[i][j])
            else:
                P1[i][j] = P[i][j]*(1-Rho[i][j])
    prob2 = P1.sum()/(n*(n-1)) # calculate the probability of 1 shown in G2
    
    # generate a Erdos-Renyi graph G2 based on the correlation matrix
    G2 = sample_edges(P1, directed = False, loops = False) 
    G2 = G2 - np.diag(np.diag(G2))
    prob3 = G2.sum()/(n*(n-1))
    return G2
