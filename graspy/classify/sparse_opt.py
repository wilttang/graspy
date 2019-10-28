# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numba import njit

from .base import BaseClassify


class SparseOpt(BaseClassify):
    """
    Network classification algorithm using sparse optimization.
    """
    def __init__(self, lambda_=0, gamma=1e-5, nodes=None):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.nodes = nodes

    def _admm(self, omega1, omega2, tol=1e-7, beta_start=None):
        n = self.y.size
        m = self.D.shape[0]

        if not beta_start:
            beta = self.y
        else:
            beta = beta_start

        q = _soft_threshold(self.y, omega1)
        r = self.D @ q
        if np.max(np.absolute(q)) == 0:
            beta = q
            i = 0
            conv_crit = 0
            best_beta = q

            self.beta = beta
            self.q = q
            self.r = r
            self.iter = i
            self.conv_crit = conv_crit
            self.best_beta=best_beta

            return beta, q, r, i, conv_crit, best_beta

        qk = q
        rk = r
        u = np.zeros(n)
        v = np.zeros(m)
        i = 0
        phi_beta_k = (0.5*np.sum(self.y-beta) + omega1*np.sum(np.abs(beta)) 
                      + omega2*_gl_penalty(self.D @ beta))
        conv_crit = np.infty
        sk = np.infty
        resk = np.infty
        best_beta = beta
        best_phi = phi_beta_k

        while (resk > tol or sk > tol) and i <= self.max_iter:
            aux = self.y-u + self.rho*q - self.D @ (self.rho*r - v)
            beta = aux / (1+3*self.rho)
            Dbeta = self.D @ beta

            # update q
            q = _soft_threshold(beta+u/self.rho, omega1/self.rho)

            # update r
            Dbetavrho = Dbeta + v/self.rho
            r = _soft_threshold(Dbetavrho, omega2/self.rho)

            u = u + self.rho * (beta-q)
            v = v + self.rho * (Dbeta-r)

            # update convergence criteria
            phi_beta_k1 = (0.5*np.sum(self.y-beta)
                           + omega1*np.sum(np.abs(beta))
                           + omega2*_gl_penalty(self.D, self.nodes))
            sk = (self.rho * np.max(np.absolute(q-qk))
                  + np.max(np.absolute(np.cross(self.D, r-rk))))

            res1k = np.sqrt(np.sum(beta-q))
            res2k = np.sqrt(np.sum(Dbeta-r))

            resk = res1k + res2k
            qk = q
            rk = r
            conv_crit = np.abs(phi_beta_k1 - phi_beta_k) / phi_beta_k
            phi_beta_k = phi_beta_k1

            if phi_beta_k1 < best_phi:
                best_beta = beta
                best_phi = phi_beta_k
                break

            i += 1

        beta_q = beta * (q == 0).astype(int)
        phi_beta_q = (0.5*np.sum(self.y-beta_q)
                      + omega1*np.sum(np.abs(beta_q))
                      + omega2*_gl_penalty(self.D @ beta_q))
        whichm = np.min([phi_beta_k1, best_phi, phi_beta_q])
        if whichm == 1:
            best_beta = beta
        elif whichm == 3:
            best_beta = beta_q

        self.beta = beta
        self.q = q
        self.r = r
        self.iter = i
        self.conv_crit = conv_crit
        self.best_beta = best_beta

        return beta, q, r, i, conv_crit, best_beta

    def _logistic_lasso(self, lambda1, lambda2):
        self.rho = 1
        n = self.y.size
        p = self.x.shape[1]
        b_derivative = lambda xbeta, b: (np.sum(-self.y / (1 +
            np.exp(self.y * (xbeta+b)))) / n)
        b_hessian = lambda xbeta, b: (np.sum(1 / (np.exp(-self.y * (xbeta+b))
            + np.exp(self.y * (xbeta+b)) + 2)) / n)

        grad_f = lambda xbeta, b, beta: (-np.cross(self.x,
            self.y / (1 + np.exp(self.y * (xbeta+b)))) / n
            + self.gamma*self.beta)
        f = lambda xbeta, b, beta: (np.sum(np.log(1 + np.exp(-self.y
            * (xbeta+b)))) / n + self.gamma/2 * np.cross(beta, beta))
        penalty = lambda beta: (lambda1*np.sum(np.abs(beta))
            + lambda2*_gl_penalty(self.D@beta, self.nodes))

        def proximal(u, lambda_, beta_startprox=None, tol=1e-7):
            if lambda2 > 0:
                gl = self._admm(omega1=lambda_*lambda1,
                    omega2=lambda_*lambda2, beta_start=beta_startprox)
                return gl[5], gl[1], gl[2]
            elif lambda1 > 0:
                return np.sign(np.max(np.abs(u) - (lambda1*lambda_), 0))
            else:
                return u

        def b_step(xbeta, b_start=0):
            tolb = 1e-4
            max_sb = 100
            b_n = b_start
            i = 0
            b_deriv = np.inf
            while np.abs(b_derivative) > tolb and i < max_sb:
                b_deriv = b_deriv(xbeta, b_n)
                b_hess = b_hessian(xbeta, b_n)
                b_n = b_deriv/(b_hess + 1 * (np.abs(b_deriv/b_hess) > 100))
                i += 1
            return b_n

        beta_start = np.zeros(p)
        b_start = 0

        optimal = self._fista(proximal, b_step, f, grad_f, penalty,
                              beta_start, b_start)

        return optimal

    def _fista(self, proximal, b_step, f, grad_f, penalty, beta_start,
               b_start):
        xk1 = self.x
        xk = self.x
        criterion = np.infty
        crit_f = np.infty

        k = 0
        tk = 0.125
        beta_step = 0.5

        xbeta = x @ x_start
        b = b_step(xbeta, b_start)
        optimal = [xk1, xk, criterion, crit_f, k, tk, beta_step, b]

        return optimal

    def _predict(self):
        pass

    def fit(self, x, y, xtest):
        self.nodes = (1 + np.sqrt(1 + 8*x.shape[1])) / 2
        yorig = y
        self.y = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1

        alpha_norm = np.std(x)
        self.x = x / alpha_norm
        lambda1 = self.lambda_ * self.rho / alpha_norm
        lambda2 = self.lambda_ / alpha_norm
        gamma = self.gamma
        self.D = _construct_d(self.nodes)

        self.beta_start = np.zeros(self.nodes*(self.nodes-1)/2)
        self.b_start = 0
        self.max_iter = 300
        self.conv_crit = 1e-5
        self.max_time = np.infty

        self._logistic_lasso(lambda1, lambda2)
        beta = self.best_beta / alpha_norm
        b = self.best_b
        yfit = alpha_norm * (self.x @ beta)
        self.yfit = np.exp(yfit) / (1 + np.exp(yfit))
        self.train_error = 1 - np.sum(np.diag(yfit)) / len(self.y)
        self._predict()
        yfit_test = xtest @ beta + b
        self.yfit_test = np.exp(yfit_test) / (1 + np.exp(yfit_test))
        self.test_error = 1 - np.sum(np.diag(yfit_test)) / len(yfit_test)

        return self


@njit
def _soft_threshold(x, lambda_, nodes=None):
    n = x.shape[0]

    for i in range(n):
        norm_node = _gl_penalty(x, nodes)
        t = 1 - lambda_/norm_node
        for j in range((nodes-1)*i, (nodes-1)*(i+1)):
            if norm_node <= lambda_:
                x[j] = 0
            else:
                x[j] *= t

    return x

@njit
def _gl_penalty(b, nodes=None):
    for i in range(nodes):
        norm_node = np.sum([b[j] ** 2 for j in range((nodes-1)*i, (nodes-1)*(i+1))])
        gl = np.sqrt(norm_node)

    return gl


@njit
def _construct_d(nodes=264):
    B = np.zeros(shape=(nodes, nodes))
    B[np.triu_indices(nodes, k=1)] = [i * (i+1) / 2 for i in range(nodes)]
    B = B + B.T
    D = np.zeros(shape=(nodes*(nodes-1), nodes*(nodes-1)/2))
    for i in range(nodes):
        D[i*(nodes-1) + 1:(i+1)*(nodes-1) + 1, B[i, -i]] = np.diag(nodes-1)

    return D