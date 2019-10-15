import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from er_corr1 import er_corr


def figure_p(n, p, rho):
    global origin_G1, G1, m_org

    G1, G2 = er_corr(10, 0.5, 0.3)
    origin_G1, G1 = er_corr(n, p, rho)
    # m_org = origin_G1.mean()
    m_gen = G1.mean()
    return m_gen


if __name__ == '__main__':
    n = 1000
    prob = []
    for i in range(n):
        prob.append(figure_p(100, 0.5, 0.3))
        prob.sort()
    print(prob)
    plt.draw()
    plt.plot(prob)
    # plt.show()
    print(np.median(prob))
    print(np.mean(prob))

