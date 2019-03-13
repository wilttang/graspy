import numpy as np
np.random.seed(88889999)
import graspy
from graspy.inference import SemiparametricTest
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import sbm, rdpg
from graspy.utils import symmetrize
from graspy.plot import heatmap, pairplot

import numpy as np
from mgcpy.independence_tests.dcorr import DCorr
from sklearn import preprocessing
from mgcpy.independence_tests.mgc.mgc import MGC
import math
from tqdm import tqdm


def permute_matrix(A):
    permuted_indices = np.random.permutation(np.arange(A.shape[0]))
    A = A[np.ix_(permuted_indices, permuted_indices)]
    return A


def ident(x):
    return x


def power(indept_test, sample_func, transform_func, mc=100, alpha=0.05,
          is_null=False, **kwargs):
    test_stat_null_array = np.zeros(mc)
    test_stat_alt_array = np.zeros(mc)
    for i in range(mc):
        A, B = sample_func(**kwargs)
        if is_null:
            A = permute_matrix(A)
        test_stat_alt, _ = indept_test.test_statistic(
            matrix_X=transform_func(A), matrix_Y=transform_func(B), is_fast=True)
        test_stat_alt_array[i] = test_stat_alt

        # generate the null by permutation
        A_null = permute_matrix(A)
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=transform_func(A_null), matrix_Y=transform_func(B), is_fast=True)
        test_stat_null_array[i] = test_stat_null
    # if pearson, use the absolute value of test statistic then use one-sided
    # rejection region
    if indept_test.get_name() == 'pearson':
        test_stat_null_array = np.absolute(test_stat_null_array)
        test_stat_alt_array = np.absolute(test_stat_alt_array)
    critical_value = np.sort(test_stat_null_array)[math.ceil((1-alpha)*mc)]
    power = np.where(test_stat_alt_array > critical_value)[0].shape[0] / mc
    return power


def get(n=50):
    X1 = np.random.uniform(0.2,0.7,n).reshape(-1,1)
    # X2 = np.random.uniform(0.2,0.7,n).reshape(-1,1)
    X2 = np.random.uniform(0.4,0.9,n).reshape(-1,1)
    A1 = rdpg(X1,
          loops=False,
          rescale=False,
          directed=False)
    A2 = rdpg(X2,
          loops=False,
          rescale=False,
          directed=False)
    return A1, A2


# NOT USED RN
def compute_mgc(X, Y):
    mgc = MGC()
    fast_mgc_data={"sub_samples": 10} # fast mgc data preparation
    fast_mgc_statistic, test_statistic_metadata = mgc.test_statistic(X, Y, is_fast=True, fast_mgc_data=fast_mgc_data)
    #mgc_statistic, independence_test_metadata = mgc.test_statistic(X, Y)
    p_value, metadata = mgc.p_value(X, Y)

    # print("MGC test statistic:", mgc_statistic)
    # print("P Value:", p_value)
    # print("Optimal Scale:", independence_test_metadata["optimal_scale"])
    return fast_mgc_statistic, p_value, test_statistic_metadata # independence_test_metadata
# NOT USED RN


pows = []
x = range(10,101,10)
for _ in range(10):
    xs = []
    for n in tqdm(x):
        mgc = MGC()
        #dcorr = DCorr()
        p = power(mgc, get, ident, n=n) #paired_
        xs.append(p)
    pows.append(xs)


import matplotlib.pyplot as plt
for i in range(10):
    plt.plot(x,pows[i], 'b-.', alpha=0.4)
plt.plot(x,[0.05]*len(x),'r-.',alpha=0.8)
plt.xlabel('n')
plt.ylabel('power')
plt.savefig('power_curve_mgc_alt.png')
plt.show()
