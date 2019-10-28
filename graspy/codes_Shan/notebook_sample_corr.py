#%%
import numpy as np
from graspy.simulations import sample_edges
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import copy
import warnings
import sample_corr

#%%
#definite simulated P and Rho matrices
p = 0.5
rho = 0.3
n = 100
P = p * np.ones( (n,n) )
Rho = rho * np.ones((n,n))
#%%
def parameters_sample_corr(P, Rho, directed=False, loops=False):
    A = sample_corr(P, Rho, directed=False, loops=False)
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
#%%
# tests for properties of sample_corr function

# return prob of G2 without diagnal elements
def test1_sample_corr(P, Rho, directed=False, loops=False):
    G1 = sample_edges(P, directed = False, loops = False)
    origin_G1 = copy.deepcopy(G1)
    prob1 = origin_G1.sum()/(n*(n-1))

    P1 = copy.deepcopy(P)
    Rho = copy.deepcopy(Rho)
    for i in range(n):
        for j in range(n):
            if G1[i][j] == 1:
                P1[i][j] = P[i][j]+Rho[i][j]*(1-P[i][j])
            else:
                P1[i][j] = P[i][j]*(1-Rho[i][j])
    prob2 = P1.sum()/(n*(n-1))
    
    G2 = sample_edges(P1, directed = False, loops = False)
    G2 = G2 - np.diag(np.diag(G2))
    prob3 = G2.sum()/(n*(n-1))
    return prob3
#%%
# show the PDF of probability of G2
p=[]
k=0
sum=0
for i in range (800):
    k = test1_sample_corr(P, Rho, directed=False, loops=False)
    p.append(k)
p.sort()
m = np.mean(p)
print(f'mean of revised probabilities of graph2 is ',m)

%matplotlib inline
k=0.5
#plt.xlim((0.45, 0.56))
#plt.ylim((0, 55))
sns.distplot(p,axlabel='probability values of p', kde_kws={"label":"density curve of revised probabilities of '1' in graph2","color":"white"})
plt.bar(k, 70, width=0.001, alpha = 0.7, color='blue',label='expacted probability')
plt.bar(m, 70, width=0.001, alpha = 0.7, color='yellow',label='the mean value of probabilities of G2')
plt.legend(loc='upper left')
plt.show()

# %%
#definite simulated P and Rho matrices
p = 0.5
rho = 0.3
n = 100
P = p * np.ones( (n,n) )
Rho = rho * np.ones((n,n))

# return origin G1 and generated G2
def sample_corr_correlation(P, Rho, directed=False, loops=False):
    G1 = sample_edges(P, directed = False, loops = False)
    origin_G1 = copy.deepcopy(G1)
    prob1 = origin_G1.sum()/n**2

    P1 = copy.deepcopy(P)
    Rho = copy.deepcopy(Rho)
    for i in range(n):
        for j in range(n):
            if G1[i][j] == 1:
                P1[i][j] = P[i][j]+Rho[i][j]*(1-P[i][j])
            else:
                P1[i][j] = P[i][j]*(1-Rho[i][j])
    prob2 = P1.sum()/n**2
    
    G2 = sample_edges(P1, directed = False, loops = False)
    G2 = G2 - np.diag(np.diag(G2))
    prob3 = G2.sum()/(n*(n-1))
    return G1, G2

# calculate the mean value of rho in graph 2
def calc_rho(freq):
    h=0
    for i in range (freq):
        g1, g2 = sample_corr_correlation(P, Rho, directed=False, loops=False)
        k=0 
        r=0
        for i in range(n):
            for j in range(n):
                if g1[i][j] == 1 and g2[i][j] == 1:
                    k+=1
        k = k/(n*(n-1))
        #print(k)
        r = np.abs((k-p**2)/(p-p**2))
        h += r
    avr = h/freq
    return avr
calc_rho(100)


# %%
