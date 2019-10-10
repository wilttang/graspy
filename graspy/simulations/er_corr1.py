from graspy.simulations import er_np
from scipy.stats import bernoulli
import copy


def er_corr(n, p, rho):
    """
    Generate a correlated Erdos-Renyi graph G2 based on G1 graph
    with Bernoulli distribution.

    Parameters
    ----------
    n: dimension if the matrix
    p: a real number in the interval (0,1)
        the probability of G1 with the Bernoulli distribution
        also a parameter for generate G2
    rho: a real number in the interval [0,1]
        another parameter to definite the correlation between graph1 and graph2

    Returns
    -------
    origin_G1: origin matrix of graph1
    G1: generated adjacency matrix of graph2

    Examples
    --------
    >>> er_corr(10, 0.5, 0.3)
    >>> print(G1)
    >>> print(G2)
    array([[0. 1. 1. 0. 0. 1. 1. 1. 0. 0.],
           [1. 0. 1. 1. 1. 1. 1. 0. 0. 0.],
           [1. 1. 0. 0. 1. 0. 0. 0. 0. 1.],
           [0. 1. 0. 0. 0. 1. 0. 1. 1. 1.],
           [0. 1. 1. 0. 0. 1. 1. 1. 0. 0.],
           [1. 1. 0. 1. 1. 0. 0. 1. 1. 0.],
           [1. 1. 0. 0. 1. 0. 0. 0. 0. 0.],
           [1. 0. 0. 1. 1. 1. 0. 0. 1. 0.],
           [0. 0. 0. 1. 0. 1. 0. 1. 0. 1.],
           [0. 0. 1. 1. 0. 0. 0. 0. 1. 0.]]
          [[0. 1. 1. 1. 1. 0. 1. 1. 0. 0.],
           [0. 1. 1. 0. 0. 0. 1. 0. 1. 0.],
           [1. 0. 0. 1. 0. 0. 0. 0. 0. 1.],
           [1. 1. 1. 0. 1. 1. 0. 1. 1. 1.],
           [0. 1. 1. 0. 1. 1. 0. 0. 0. 1.],
           [1. 0. 0. 0. 1. 1. 1. 1. 1. 0.],
           [0. 1. 1. 1. 1. 0. 1. 0. 0. 1.],
           [1. 1. 0. 1. 1. 1. 0. 0. 1. 0.],
           [0. 0. 0. 1. 0. 1. 1. 1. 1. 1.],
           [0. 1. 0. 0. 0. 1. 1. 0. 0. 0.]])
    """
    G1 = er_np(n, p)
    origin_G1 = copy.deepcopy(G1)
    sumG1 = 0
    for i in range(n):
        for j in range(n):
            sumG1 += origin_G1[i][j]

    for i in range(n):
        for j in range(n):
            if G1[i][j] == 1:
                G1[i][j] = bernoulli.rvs(p+rho*(1-p), size=1, loc=0)
            else:
                G1[i][j] = bernoulli.rvs(p*(1-rho), size=1, loc=0)
    sumG2 = 0
    for i in range(n):
        for j in range(n):
            sumG2 += G1[i][j]

    return origin_G1, G1


if __name__ == '__main__':

    G1, G2 = er_corr(10, 0.5, 0.3)
    print(G1)
    print(G2)
